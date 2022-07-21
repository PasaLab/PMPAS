# python -m benchmarks.ea_iter_ray --max_iter=1000 -d 11
import json
import logging
import os
import os.path as osp
import time

import geatpy as ea
import psutil
import ray
from sklearn.metrics import accuracy_score

from core.pdeas.model_manager.model_manager_ray import Model
from core.edeas.evolution_controller.SEGA_v1 import soea_SEGA_templet
from core.edeas.search_space.dag_space import DAGSpace
from core.edeas.search_space.cp_space import CPSpace
from core.utils.data_util import load_data_by_id
from core.utils.env_util import environment_init, ea_summary, split_validation_data, get_ea_ray_args
from core.utils.helper import plain_encoding_to_model
from core.utils.log_util import create_logger

# get logger
logger = create_logger()

# set environment
environment_init()

ray.init()

# load console args
args = get_ea_ray_args()
data = args.data
max_iter = args.max_iter
max_cell = args.max_cell
model_time_limit = args.model_time_limit
cell_time_limit = args.cell_time_limit
space = args.space
kth = args.kth
stages = args.stages

# summary of the experiment
ea_summary(data, space, max_iter)

# create directory to save the result
time_str = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time()))
RESULTS_SAVED_PATH = f"ea_ray_{space}_cell_{max_cell}_{time_str}_max_iter_{max_iter}_data_{data}_k_fold_{kth}"
cur_dir = osp.realpath(__file__)
data_dir = osp.normpath(osp.join(cur_dir, osp.pardir, osp.pardir, 'results', RESULTS_SAVED_PATH))
if not osp.exists(data_dir):
    os.makedirs(data_dir)

# load dataset
x_train, x_test, y_train, y_test = load_data_by_id(data, kth)
X_train, X_val, Y_train, Y_val = split_validation_data(x_train, y_train)

# initializing search_space instance

if space == 'DAG':
    problem = DAGSpace(X_train, Y_train, X_val, Y_val)
elif space == 'plain':
    problem = CPSpace(X_train, Y_train, X_val, Y_val)
else:
    raise ValueError("Not supported search space.")

# load dataset
x_train, x_test, y_train, y_test = load_data_by_id(data)
X_train, X_val, Y_train, Y_val = split_validation_data(x_train, y_train)

# track the best model and it's score
best_model = None
best_score = 0

# running exit flag and timer, used to break the loop and track the running time
timeout_flag = False
start_time = time.time()
"""=================================种群设置=============================="""

# population settings
Encoding = 'BG'  # 编码方式
NIND = 8  # 种群规模
Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)  # 创建区域描述器
population = ea.Population(Encoding, Field, NIND)  # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）

# evolution_controller settings
"""===============================算法参数设置============================="""
myAlgorithm = soea_SEGA_templet(problem, population, result_path=data_dir)  # 实例化一个算法模板对象
myAlgorithm.MAXGEN = max_iter
myAlgorithm.logTras = 1  # 设置每隔多少代记录日志，若设置成0则表示不记录日志
myAlgorithm.verbose = False  # 设置是否打印输出日志信息
myAlgorithm.drawing = 0  # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画；3：绘制决策空间过程动画）
"""==========================调用算法模板进行种群进化========================"""

# run evolution_controller template
[BestIndi, population] = myAlgorithm.run()  # 执行算法模板，得到最优个体以及最后一代种群

total_time = time.time() - start_time
if BestIndi.sizes != 0:
    # get best model and get its score
    best_index = BestIndi.Phen.flatten()
    config = plain_encoding_to_model(best_index)
    best_model = Model(problem.best_model)
    best_model.fit(X_train, Y_train, X_val, Y_val)
    best_model.refit(x_train, y_train)
    best_score = accuracy_score(y_test, best_model.predict(x_test))

    # record results to a json file
    result_json = {
        'evolution_controller': 'ea',
        'search space': space,
        'ray': True,
        'stages': stages,
        'max cell': max_cell,
        'max iterations': max_iter,
        'final model': best_model.cas_model.get_child(),
        'accuracy': best_score,
        'layer depth': best_model.cas_model.best_layer_id,
        'total evaluated models': problem.model_count,
        'time cost': total_time,

    }
    logger.info(result_json)

    # dump json results to a file
    with open(osp.join(data_dir, 'result.json'), 'w') as f:
        json.dump(result_json, f)

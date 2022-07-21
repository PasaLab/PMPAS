# python -m benchmarks.ea_time_reg --max_iter=1000 -d 505
import json
import os
import os.path as osp
import time

import geatpy as ea
from sklearn.metrics import r2_score

from core.edeas.evolution_controller.SEGA_v1 import soea_SEGA_templet
from core.edeas.search_space.dag_space import DAGSpace
from core.edeas.search_space.cp_space import CPSpace
from core.utils.data_util import load_data_by_id
from core.utils.env_util import environment_init, split_validation_data, get_ea_time_args, ea_time_summary
from core.utils.log_util import create_logger

# get logger
logger = create_logger()

# set environment
environment_init()

# load console args
args = get_ea_time_args()
data = args.data
max_iter = args.max_iter
total_time = args.total_time
model_time_limit = args.model_time_limit
cell_time_limit = args.cell_time_limit
space = args.space
kth = args.kth

# summary of the experiment
ea_time_summary(data, space, total_time)

# create directory to save the result
time_str = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time()))
RESULTS_SAVED_PATH = f"ea_{space}_{time_str}_total_time_{total_time}_data_{data}_k_fold_{kth + 1}"
cur_dir = osp.realpath(__file__)
data_dir = osp.normpath(osp.join(cur_dir, osp.pardir, osp.pardir, 'results', RESULTS_SAVED_PATH))
if not osp.exists(data_dir):
    os.makedirs(data_dir)

# load dataset
x_train, x_test, y_train, y_test = load_data_by_id(data, kth, task='regression')
X_train, X_val, Y_train, Y_val = split_validation_data(x_train, y_train)

# initializing search_space instance
if space == 'DAG':
    problem = DAGSpace(X_train, Y_train, X_val, Y_val)
elif space == 'plain':
    problem = CPSpace(X_train, Y_train, X_val, Y_val)
else:
    raise ValueError("Not supported search space.")


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
myAlgorithm = soea_SEGA_templet(problem, population, result_path=data_dir, stop_by_time=True)  # 实例化一个算法模板对象
myAlgorithm.MAXGEN = max_iter
myAlgorithm.MAXTIME = total_time
myAlgorithm.logTras = 1  # 设置每隔多少代记录日志，若设置成0则表示不记录日志
myAlgorithm.verbose = False  # 设置是否打印输出日志信息
myAlgorithm.drawing = 0  # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画；3：绘制决策空间过程动画）
"""==========================调用算法模板进行种群进化========================"""

# run evolution_controller template
[BestIndi, population] = myAlgorithm.run()  # 执行算法模板，得到最优个体以及最后一代种群

if BestIndi.sizes != 0:
    # get best model and get its score
    best_index = BestIndi.Phen.flatten()
    best_model = problem.best_model
    best_model.refit(x_train, y_train)
    best_score = r2_score(y_test, best_model.predict(x_test))

    # record results to a json file
    result_json = {
        'evolution_controller': 'ea',
        'search space': space,
        'total time': total_time,
        'final model': best_model.get_child(),
        'r2': best_score,
        'time cost': time.time() - start_time,
        'layer depth': best_model.best_layer_id,
        'total evaluated models': problem.model_count
    }
    logger.info(result_json)

    # dump json results to a file
    with open(osp.join(data_dir, 'result.json'), 'w') as f:
        json.dump(result_json, f)

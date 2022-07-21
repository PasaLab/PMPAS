# python -m benchmarks.ea_iter_cv --max_iter=1000 -d 11
import json
import os
import os.path as osp
import time

import geatpy as ea
import numpy as np
from sklearn.metrics import accuracy_score

from core.edeas.evolution_controller.SEGA_v1 import soea_SEGA_templet
from core.edeas.search_space.dag_space import DAGSpace
from core.edeas.search_space.cp_space import CPSpace
from core.utils.data_util import load_data_by_id
from core.utils.env_util import environment_init, ea_summary, get_ea_args, split_validation_data
from core.utils.log_util import create_logger

# init time
init_time = time.time()

# get logger
logger = create_logger()

# set environment
environment_init()

# load console args
args = get_ea_args()
data = args.data
max_cell = args.max_cell
max_iter = args.max_iter
model_time_limit = args.model_time_limit
cell_time_limit = args.cell_time_limit
space = args.space
n_splits = args.n_splits
kth = args.kth

# summary of the experiment
ea_summary(data, space, max_iter)

# create result dir
time_str = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time()))
RESULTS_SAVED_PATH = f"ea_{space}_{time_str}_max_cell_{max_cell}_max_iter_{max_iter}_data_{data}"
cur_dir = osp.realpath(__file__)
base_dir = osp.normpath(osp.join(cur_dir, osp.pardir, osp.pardir, 'results', RESULTS_SAVED_PATH))
if not osp.exists(base_dir):
    os.makedirs(base_dir)

# record average scores
scores = []
model_counts = []
layer_depths = []
result_json = {}

for kth in range(n_splits):
    # load dataset
    x_train, x_test, y_train, y_test = load_data_by_id(data, kth)
    X_train, X_val, Y_train, Y_val = split_validation_data(x_train, y_train)

    # create current fold to save the results
    data_dir = osp.normpath(osp.join(base_dir, str(kth + 1)))
    if not osp.exists(data_dir):
        os.makedirs(data_dir)

    # initializing search_space instance
    if space == 'DAG':
        problem = DAGSpace(X_train, Y_train, X_val, Y_val, kth=kth, max_cell=max_cell)
    elif space == 'plain':
        problem = CPSpace(X_train, Y_train, X_val, Y_val, kth=kth, max_cell=max_cell)
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
    myAlgorithm = soea_SEGA_templet(problem, population, result_path=data_dir)  # 实例化一个算法模板对象
    myAlgorithm.MAXGEN = max_iter
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
        best_score = accuracy_score(y_test, best_model.predict(x_test))

        # record results to a json file
        cur_result_json = {
            'final model': best_model.get_child(),
            'accuracy': best_score,
            'time cost': time.time() - start_time,
            'layer depth': best_model.best_layer_id,
            'total evaluated models': problem.model_count
        }
        logger.info(f"k_fold: {kth + 1}.")
        logger.info(cur_result_json)
        # dump cur fold json results to a file
        with open(osp.join(data_dir, 'result.json'), 'w') as f:
            json.dump(cur_result_json, f)
        result_json[kth + 1] = cur_result_json
        scores.append(best_score)
        layer_depths.append(best_model.best_layer_id)
        model_counts.append(problem.model_count)

# record mean value of the results
cvs = np.mean(scores)
mean_layer_depth = np.mean(layer_depths)
mean_model_counts = np.mean(model_counts)
result_json['evolution_controller'] = 'ea'
result_json['search space'] = space
result_json['max iterations'] = max_iter
result_json['max cell'] = max_cell
result_json['cross val score'] = cvs
result_json['average layer depth'] = mean_layer_depth
result_json['average model counts'] = mean_model_counts
logger.info(result_json)

# dump final json results to a file
with open(osp.join(base_dir, 'result.json'), 'w', encoding='utf-8') as f:
    json.dump(result_json, f)

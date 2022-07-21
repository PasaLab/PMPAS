# python -m benchmarks.bdas_cell -s DAG -B 8 -K 8 -d 23
import csv
import json
import os
import os.path as osp
import platform
import time

import numpy as np
import psutil
from sklearn.metrics import accuracy_score
from timeout_decorator import TimeoutError

from core.pdeas.proxy_model import ProxyModel
from core.pdeas.model_manager.model_manager_classification import Model
from core.pdeas.search_space import DAGSpace, CPSpace
from core.utils.data_util import load_data_by_id
from core.utils.env_util import environment_init, split_validation_data, get_bdas_cell_args
from core.utils.log_util import create_logger


def summary():
    """
    Summary of the configuration of the experiment.
    :return:
    """
    GB = 2 ** 30

    logger.info(f"BDAS on {data}, search space: {space}, K: {K}, max cell: {max_cell}, "
                f"estimators {estimators}, "
                f"time limit for per cell: {cell_time_limit}, "
                f"time limit for per model: {model_time_limit}.")
    logger.info(f"""{'*' * 10}Experiment environment{'*' * 10}
    platform: {platform.platform()}"
    architecture: {platform.architecture()}
    processor: {platform.processor()}
    physical cores: {psutil.cpu_count(logical=False)}
    virtual cores: {psutil.cpu_count()}
    total memory: {psutil.virtual_memory().total / GB:.2f}G
    available memory: {psutil.virtual_memory().available / GB:.2f}G
    python version: {platform.python_version()}
    process id: {os.getpid()}.\n""")


# get the logger to logging experiments
logger = create_logger()

# set environment
environment_init()

# load console args
args = get_bdas_cell_args()
# todo,more beauty please
data, K, space, model_time_limit, cell_time_limit, estimators, exclude_estimators = \
    args.data, args.Kmost, args.space, args.model_time_limit, \
    args.cell_time_limit, args.estimators, args.exclude_estimators
max_cell = args.max_cell
kth = args.kth
strategy = args.strategy

# exclude estimators
if exclude_estimators is not None:
    for exclude_estimator in exclude_estimators:
        try:
            estimators.remove(exclude_estimator)
        except ValueError:
            raise ValueError("Current estimator not supported.")

# create directory to save the result
time_str = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
RESULTS_SAVED_PATH = f"bdas_{time_str}_cell_{max_cell}_data_{data}"
cur_dir = osp.realpath(__file__)
data_dir = osp.normpath(osp.join(cur_dir, osp.pardir, osp.pardir, 'results', RESULTS_SAVED_PATH))
if not osp.exists(data_dir):
    os.makedirs(data_dir)

# initializing search space and controller
if space == 'DAG':
    sp = DAGSpace(max_cell, estimators)
else:
    sp = CPSpace(max_cell, estimators)
controller = ProxyModel(sp, K, saved_path=data_dir, strategy=strategy)

# load dataset
x_train, x_test, y_train, y_test = load_data_by_id(data, kth)
X_train, X_val, Y_train, Y_val = split_validation_data(x_train, y_train)

# summary of the experiment
summary()

# track the best model and it's score
best_model = None
best_score = 0
model_count = 0

# record running time
start_time = time.time()

# record average score and best score
cell_average_scores = []
cell_best_scores = []

# begin the growing process
for trial in range(max_cell):
    if trial == 0:
        k = None
    else:
        k = K
    logger.info(f"Begin cell number = {trial + 1}")
    # get all the model for the current trial
    actions = controller.get_models(top_k=k)
    rewards = []
    for t, action in enumerate(actions, 1):
        logger.info(f"Current #{t} / #{len(actions)}.")
        logger.info(action)
        cc = Model(cas_config=action, model_time_limit=model_time_limit, cell_time_limit=cell_time_limit)
        try:
            cc.fit(X_train, Y_train, X_val, Y_val, confidence_screen=True)
        except TimeoutError:
            logger.info("Time limit for the current model has reached.")
            logger.info(f"#{model_count}, score: 0.\n")
            model_count += 1
            rewards.append(0)
            continue

        except:
            logger.info(f"{action} running failed on dataset: {data}.")
            logger.info(f"#{model_count}, score: 0.\n")
            model_count += 1
            rewards.append(0)
            continue
        model_count += 1
        reward = cc.cas_model.best_score
        # record the best model
        if reward > best_score:
            best_model = cc
            best_score = reward
        logger.info(f"#{model_count}, score: {reward}.\n")
        rewards.append(reward)
        # write the results of this trial into a file
        with open(osp.join(data_dir, 'train_history.csv'), mode='a+', newline='') as f:
            current_data = [reward]
            current_data.extend(action)
            writer = csv.writer(f)
            writer.writerow(current_data)
    # train and update controller
    loss = controller.finetune_proxy_model(rewards)
    controller.update_search_space()
    trial += 1
    logger.info(f"Loss of the current cell number {trial}: {loss}.\n")

    cell_average_scores.append(np.mean(rewards))
    cell_best_scores.append(np.max(rewards))

logger.info("Maximum cell has reached.\n")

# refit and score
logger.info(f"Search process finished, begin to refit {best_model.cas_model.get_child()} on {data}.")
best_model.refit(x_train, y_train)
final_score = accuracy_score(y_test, best_model.predict(x_test))
logger.info(
    f"BDAS on {data}, "
    f"search space: {space}, "
    f"strategy: {strategy}, "
    f"K: {K}, "
    f"max cell: {max_cell}, "
    f"time limit for per cell: {cell_time_limit}, "
    f"time limit for per model: {model_time_limit}, "
    f"final model: {best_model.cas_model.get_child()}, "
    f"score: {final_score}, "
    f"time cost: {time.time() - start_time}, "
    f"layer depth: {best_model.cas_model.best_layer_id}, "
    f"total evaluated models: {model_count}.")

# save to files
# data average_score best score
result_json = {}
result_json['evolution_controller'] = 'pdeas'
result_json['ray'] = True
result_json['search space'] = space
result_json['max cell'] = max_cell
result_json['strategy'] = strategy
result_json['K'] = K
result_json['kth fold'] = kth + 1
result_json['cell time limit'] = cell_time_limit
result_json['model time limit'] = model_time_limit
result_json['final model'] = best_model.cas_model.get_child()
result_json['score'] = final_score
result_json['layer depth'] = best_model.cas_model.best_layer_id
result_json['total evaluated models'] = model_count
result_json['time cost'] = time.time() - start_time

logger.info(result_json)

# dump json results to a file
with open(osp.join(data_dir, 'result.json'), 'w') as f:
    json.dump(result_json, f)

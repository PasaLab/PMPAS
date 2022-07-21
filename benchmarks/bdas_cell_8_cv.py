# python -m benchmarks.bdas_cell_cv_val -d 23 --max_cell 8

import csv
import os
import os.path as osp
import platform
import time
import json
from collections import defaultdict

import numpy as np
import psutil
from timeout_decorator import TimeoutError

from core.pdeas.proxy_model import ProxyModel
from core.pdeas.model_manager.model_manager_classification import Model
from core.pdeas.search_space import DAGSpace, CPSpace
from core.utils.data_util import load_data_by_id
from core.utils.env_util import environment_init, get_bdas_cell_cv_args, split_validation_data
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

# must add, environment init
environment_init()

# init time line
init_time = time.time()

# load console args
args = get_bdas_cell_cv_args()
# todo,more beauty please
data, K, space = args.data, args.Kmost, args.space
max_cell = args.max_cell
model_time_limit, cell_time_limit, = args.model_time_limit, args.cell_time_limit
estimators, exclude_estimators = args.estimators, args.exclude_estimators
n_splits = args.n_splits
strategy = args.strategy

# exclude estimators
if exclude_estimators is not None:
    for exclude_estimator in exclude_estimators:
        try:
            estimators.remove(exclude_estimator)
        except ValueError:
            raise ValueError("Current estimator not supported.")

# show summary of the environment
summary()

scores = []
model_counts = []
layer_depths = []

# create directory to save the result
time_str = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
RESULTS_SAVED_PATH = f"bdas_{time_str}_cell_{max_cell}_data_{data}"
cur_dir = osp.realpath(__file__)
data_dir = osp.normpath(osp.join(cur_dir, osp.pardir, osp.pardir, 'results', RESULTS_SAVED_PATH))
if not osp.exists(data_dir):
    os.makedirs(data_dir)

result_json = defaultdict(dict)
result_json.update({
    'data': data,
    'search space': space,
    'K': K,
    'max cell': max_cell,
    "time limit for per cell": cell_time_limit,
    "time limit for per model": model_time_limit,
})
for kth in range(n_splits):
    # load dataset
    x_train, x_test, y_train, y_test = load_data_by_id(data, kth)
    X_train, X_val, Y_train, Y_val = split_validation_data(x_train, y_train)

    # initializing search space and controller
    if space == 'DAG':
        sp = DAGSpace(max_cell, estimators)
    else:
        sp = CPSpace(max_cell, estimators)

    # initializing controller
    controller = ProxyModel(sp, K, saved_path=data_dir, strategy=strategy)

    # track the best model and it's score
    best_model = None
    best_score = 0
    model_count = 0

    # record running time
    start_time = time.time()

    # begin the growing process
    cur_json = {}
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
            logger.info(f"{kth + 1}-fold #{t} / #{len(actions)}.")
            logger.info(action)
            cc = Model(action)
            try:
                cc.fit(X_train, Y_train, X_val, Y_val, confidence_screen=True)
            except TimeoutError:
                logger.info("Time limit for the current model has reached.")
                logger.info(f"{kth + 1}-fold #{model_count}, score: 0.\n")
                model_count += 1
                rewards.append(0)
                continue

            except:
                logger.info(f"{action} running failed on dataset: {data}.")
                logger.info(f"{kth + 1}-fold #{model_count}, score: 0.\n")
                model_count += 1
                rewards.append(0)
                continue
            model_count += 1
            reward = cc.cas_model.best_score
            # record the best model
            if reward > best_score:
                best_model = cc
                best_score = reward
            logger.info(f"{kth + 1}-fold #{model_count}, score: {reward}.\n")
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
    logger.info("Maximum cell has reached.\n")
    logger.info(f"{kth + 1}-fold process finished, begin to refit...")
    acc = best_model.cas_model.best_score
    logger.info(
        f"{kth + 1}-fold on {data}, "
        f"final model: {best_model.cas_model.get_child()}, "
        f"score: {acc}, "
        f"time cost: {time.time() - start_time}, "
        f"layer depth: {best_model.cas_model.best_layer_id}, "
        f"total evaluated models: {model_count}.\n")
    cur_json['time cost'] = time.time() - start_time
    cur_json['score'] = acc
    cur_json['layer depth'] = best_model.cas_model.best_layer_id
    cur_json['total evaluated models'] = model_count
    cur_json['final model'] = best_model.cas_model.get_child()
    result_json[kth + 1] = cur_json
    scores.append(acc)
    model_counts.append(model_count)
    layer_depths.append(best_model.cas_model.best_layer_id)
cvs = np.mean(scores)
mean_layer_depth = np.mean(layer_depths)
mean_model_counts = np.mean(model_counts)

result_json.update({'cross val score': cvs,
                    'total time cost': time.time() - init_time
                    })
logger.info(result_json)

# dump json results to a file
with open(osp.join(data_dir, 'result.json'), 'w') as f:
    json.dump(result_json, f)

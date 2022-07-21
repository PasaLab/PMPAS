import csv
import json
import os
import os.path as osp
import platform
import time

import psutil
from sklearn.metrics import accuracy_score
from timeout_decorator import TimeoutError

from core.pdeas.proxy_model import ProxyModel
from core.pdeas.model_manager.model_manager_classification import Model
from core.pdeas.search_space import DAGSpace, CPSpace
from core.rdeas.random_search import RandomSearch
from core.utils.data_util import load_data_by_id
from core.utils.env_util import environment_init, split_validation_data, get_bdas_time_args, get_random_search_args
from core.utils.log_util import create_logger

def summary():
    """
    Summary of the configuration of the experiment.
    :return:
    """
    GB = 2 ** 30


    logger.info(f"Random Search on {data}, "
                f"search space: {space}, "
                f"max cell: {max_cell}, "
                f"total time: {total_time}, "
                f"on {kth + 1}-th fold, "
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

# load console args
args = get_random_search_args()
# todo,more beauty please
data = args.data
max_cell = args.max_cell
total_time = args.total_time
space = args.space
model_time_limit = args.model_time_limit
cell_time_limit = args.cell_time_limit
estimators = args.estimators
exclude_estimators = args.exclude_estimators
kth = args.kth

# exclude estimators
if exclude_estimators is not None:
    for exclude_estimator in exclude_estimators:
        try:
            estimators.remove(exclude_estimator)
        except ValueError:
            raise ValueError("Current estimator not supported.")

# create directory to save the result
time_str = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time()))
RESULTS_SAVED_PATH = f"rs_{time_str}_time_{total_time}_data_{data}_fold{kth + 1}"
cur_dir = osp.realpath(__file__)
data_dir = osp.normpath(osp.join(cur_dir, osp.pardir, osp.pardir, 'results', RESULTS_SAVED_PATH))
if not osp.exists(data_dir):
    os.makedirs(data_dir)

# initializing search space and controller
rs = RandomSearch(space, max_cell)

# load dataset
x_train, x_test, y_train, y_test = load_data_by_id(data, kth)
X_train, X_val, Y_train, Y_val = split_validation_data(x_train, y_train)

# summary of the experiment
summary()

# track the best model and it's score
best_model = None
best_score = 0
model_count = 0
rewards = []

# running timer, used to break the loop and track the running time
start_time = time.time()

while True:
    if time.time() - start_time > total_time:
        logger.info("Total time budget has reached.")
        break
    cur_model = rs.next()
    logger.info(f"Current #{model_count}.")
    logger.info(cur_model)
    cc = Model(cas_config=cur_model, model_time_limit=model_time_limit, cell_time_limit=cell_time_limit)
    try:
        cc.fit(X_train, Y_train, X_val, Y_val, confidence_screen=True)
    except TimeoutError:
        logger.info("Time limit for the current model has reached.")
        logger.info(f"#{model_count}, score: 0.\n")
        model_count += 1
        rewards.append(0)
        continue

    except:
        logger.info(f"{cur_model} running failed on dataset: {data}.")
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
        current_data.extend(cur_model)
        writer = csv.writer(f)
        writer.writerow(current_data)

    if time.time() - start_time > total_time:
        logger.info("Total time budget has reached.")
        break

# refit and score
logger.info(f"Search process finished, begin to refit {best_model.cas_model.get_child()} on {data}.")
best_model.refit(x_train, y_train)
final_score = accuracy_score(y_test, best_model.predict(x_test))

# record results to a json file
result_json = {
    'evolution_controller': 'random search',
    'search space': space,
    'total time': total_time,
    'k-fold': kth + 1,
    'time limit for per cell': cell_time_limit,
    'time limit for per model': model_time_limit,
    'final model': best_model.cas_model.get_child(),
    'score': final_score,
    'time cost': time.time() - start_time,
    'layer depth': best_model.cas_model.best_layer_id,
    'total evaluated models': model_count
}
logger.info(result_json)

# dump json results to a file
with open(osp.join(data_dir, 'result.json'), 'w') as f:
    json.dump(result_json, f)

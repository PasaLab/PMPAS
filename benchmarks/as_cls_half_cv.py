import json
import os
import os.path as osp
import time

import numpy as np
from autosklearn.estimators import AutoSklearnClassifier
from sklearn.metrics import accuracy_score

from core.utils.data_util import load_data_by_id
from core.utils.env_util import environment_init, get_autosklearn_cls_args
from core.utils.log_util import create_logger

# create logger to record result
logger = create_logger()

# must add, environment init
environment_init()

# load parsed arguments
args = get_autosklearn_cls_args()
total_time = args.total_time = 1800
id_ = args.data
n_splits = args.n_splits

# create directory to save the result
time_str = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time()))
RESULTS_SAVED_PATH = f"autosklearn_cls_{time_str}_time_{total_time}_data_{id_}"
cur_dir = osp.realpath(__file__)
data_dir = osp.normpath(osp.join(cur_dir, osp.pardir, osp.pardir, 'results', RESULTS_SAVED_PATH))
if not osp.exists(data_dir):
    os.makedirs(data_dir)

# track scores
scores = []

# n_splits cross validation
for kth in range(n_splits):
    x_train, x_test, y_train, y_test = load_data_by_id(id_, kth)
    regressor = AutoSklearnClassifier(time_left_for_this_task=total_time, per_run_time_limit=120,
                                      initial_configurations_via_metalearning=0,
                                      resampling_strategy="cv", resampling_strategy_arguments={'folds': 3},
                                      ensemble_size=1, smac_scenario_args={"initial_incumbent": "RANDOM"})
    regressor.fit(x_train, y_train)
    y_pred = regressor.predict(x_test)
    score = accuracy_score(y_test, y_pred)
    scores.append(score)
    logger.info(f"{kth + 1}-th fold score: {score}.")
    print(f"{kth + 1}-th fold score: {score}.")

# calculate mean score
cvs = np.mean(scores)

# record results to a json file
result_json = {
    'evolution_controller': 'auto-sklearn',
    'total time': total_time,
    'score': cvs,
}

# dump json results to a file
with open(osp.join(data_dir, 'result.json'), 'w') as f:
    json.dump(result_json, f)

logger.info(f"autosklearn on {id_}, {n_splits}-fold cross val score: {cvs}, time budget: {total_time}.")
print(f"autosklearn on {id_}, {n_splits}-fold cross val score: {cvs}, time budget: {total_time}.")


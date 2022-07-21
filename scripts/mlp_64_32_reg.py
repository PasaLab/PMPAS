# python -m benchmarks.gcforest -d 37

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor

from core.utils.constants import DATASET_IDS, REGRESSION_IDS
from core.utils.data_util import load_data_by_id
from core.utils.env_util import environment_init, get_MLP_args
from core.utils.log_util import create_logger

# create logger to record result
logger = create_logger()

# must add, environment init
environment_init()

# load parsed arguments
args = get_MLP_args()
id_, n_splits, hidden_layer_sizes = args.data, args.n_splits, args.hidden_layer_sizes

df = pd.DataFrame(index=REGRESSION_IDS)
id_to_score = {}

for id_ in DATASET_IDS:
    logger.info(f"Running on dataset {id_}...")
    # track scores
    scores = []
    # n_splits cross validation
    for kth in range(n_splits):
        x_train, x_test, y_train, y_test = load_data_by_id(id_, kth, task='regression')
        clf = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        acc = r2_score(y_test, y_pred)
        scores.append(acc)
        logger.info(f"{kth + 1}-th fold score: {acc}.")
    cvs = np.mean(scores)
    logger.info(f"MLP on {id_}, {n_splits}-fold cross val score: {cvs}.")
    id_to_score[id_] = cvs
df['mlp'] = pd.Series(id_to_score)
df.to_csv('mlp_64_32_regression.csv')

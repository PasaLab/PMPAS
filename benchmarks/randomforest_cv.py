import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from core.utils.data_util import load_data_by_id
from core.utils.env_util import environment_init, get_randomforest_args
from core.utils.log_util import create_logger

# create logger to record result
logger = create_logger()

# must add, environment init
environment_init()

# load parsed arguments
args = get_randomforest_args()
id_, n_splits = args.data, args.n_splits

# track scores
scores = []

# n_splits cross validation
for kth in range(n_splits):
    x_train, x_test, y_train, y_test = load_data_by_id(id_, kth)
    clf = RandomForestClassifier(n_estimators=500)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    scores.append(acc)
    logger.info(f"{kth + 1}-th fold score: {acc}.")
cvs = np.mean(scores)
logger.info(f"Randomforest on {id_}, {n_splits}-fold cross val score: {cvs}.")

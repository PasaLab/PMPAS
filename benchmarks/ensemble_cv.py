import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from core.utils.data_util import load_data_by_id
from core.utils.env_util import environment_init, get_ensemble_args
from core.utils.log_util import create_logger

logger = create_logger()

environment_init()
args = get_ensemble_args()
id_ = args.data

n_splits = 5
df = pd.DataFrame()

# adaboost
id_to_score = {}

logger.info(f"Running on dataset {id_}...")
# track scores
scores = []
# n_splits cross validation
for kth in range(n_splits):
    x_train, x_test, y_train, y_test = load_data_by_id(id_, kth)
    clf = AdaBoostClassifier()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    scores.append(acc)
    logger.info(f"{kth + 1}-th fold score: {acc}.")
cvs = np.mean(scores)
logger.info(f"Adaboost on {id_}, {n_splits}-fold cross val score: {cvs}.")
id_to_score[id_] = cvs
df['adaboost'] = pd.Series(id_to_score)

# xgboost
id_to_score = {}
logger.info(f"Running on dataset {id_}...")
# track scores
scores = []
# n_splits cross validation
for kth in range(n_splits):
    x_train, x_test, y_train, y_test = load_data_by_id(id_, kth)
    if len(np.unique(y_train)) > 2:
        objective = 'multi:softprob'
    else:
        objective = 'binary:logistic'
    clf = XGBClassifier(objective=objective)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    scores.append(acc)
    logger.info(f"{kth + 1}-th fold score: {acc}.")
cvs = np.mean(scores)
logger.info(f"Xgboost on {id_}, {n_splits}-fold cross val score: {cvs}.")
id_to_score[id_] = cvs
df['xgboost'] = pd.Series(id_to_score)

# gbdt
id_to_score = {}
logger.info(f"Running on dataset {id_}...")
# track scores
scores = []
# n_splits cross validation
for kth in range(n_splits):
    x_train, x_test, y_train, y_test = load_data_by_id(id_, kth)
    clf = GradientBoostingClassifier()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    scores.append(acc)
    logger.info(f"{kth + 1}-th fold score: {acc}.")
cvs = np.mean(scores)
logger.info(f"GBDT on {id_}, {n_splits}-fold cross val score: {cvs}.")
id_to_score[id_] = cvs
df['gbdt'] = pd.Series(id_to_score)

# randomforest
id_to_score = {}
logger.info(f"Running on dataset {id_}...")
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
id_to_score[id_] = cvs
df['randomforest'] = pd.Series(id_to_score)

# lightgbm
id_to_score = {}
logger.info(f"Running on dataset {id_}...")
# track scores
scores = []
# n_splits cross validation
for kth in range(n_splits):
    x_train, x_test, y_train, y_test = load_data_by_id(id_, kth)
    clf = LGBMClassifier()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    scores.append(acc)
    logger.info(f"{kth + 1}-th fold score: {acc}.")
cvs = np.mean(scores)
logger.info(f"Lightgbm on {id_}, {n_splits}-fold cross val score: {cvs}.")
id_to_score[id_] = cvs
df['lightgbm'] = pd.Series(id_to_score)

df.to_csv('ensemble.csv')

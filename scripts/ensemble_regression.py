import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

from core.utils.constants import REGRESSION_IDS
from core.utils.data_util import load_data_by_id
from core.utils.env_util import environment_init
from core.utils.log_util import create_logger

logger = create_logger()

environment_init()

n_splits = 5
df = pd.DataFrame(index=REGRESSION_IDS)

# adaboost
id_to_score = {}
for id_ in REGRESSION_IDS:
    logger.info(f"Running on dataset {id_}...")
    # track scores
    scores = []
    # n_splits cross validation
    for kth in range(n_splits):
        x_train, x_test, y_train, y_test = load_data_by_id(id_, kth, task='regression')
        clf = AdaBoostRegressor()
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        score = r2_score(y_test, y_pred)
        scores.append(score)
        logger.info(f"{kth + 1}-th fold score: {score}.")
    cvs = np.mean(scores)
    logger.info(f"Adaboost on {id_}, {n_splits}-fold cross val score: {cvs}.")
    id_to_score[id_] = cvs
df['adaboost'] = pd.Series(id_to_score)

# xgboost
id_to_score = {}
for id_ in REGRESSION_IDS:
    logger.info(f"Running on dataset {id_}...")
    # track scores
    scores = []
    # n_splits cross validation
    for kth in range(n_splits):
        x_train, x_test, y_train, y_test = load_data_by_id(id_, kth, task='regression')
        clf = XGBRegressor()
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        score = r2_score(y_test, y_pred)
        scores.append(score)
        logger.info(f"{kth + 1}-th fold score: {score}.")
    cvs = np.mean(scores)
    logger.info(f"Xgboost on {id_}, {n_splits}-fold cross val score: {cvs}.")
    id_to_score[id_] = cvs
df['xgboost'] = pd.Series(id_to_score)

# gbdt
id_to_score = {}
for id_ in REGRESSION_IDS:
    logger.info(f"Running on dataset {id_}...")
    # track scores
    scores = []
    # n_splits cross validation
    for kth in range(n_splits):
        x_train, x_test, y_train, y_test = load_data_by_id(id_, kth, task='regression')
        clf = GradientBoostingRegressor()
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        score = r2_score(y_test, y_pred)
        scores.append(score)
        logger.info(f"{kth + 1}-th fold score: {score}.")
    cvs = np.mean(scores)
    logger.info(f"GBDT on {id_}, {n_splits}-fold cross val score: {cvs}.")
    id_to_score[id_] = cvs
df['gbdt'] = pd.Series(id_to_score)

# randomforest
id_to_score = {}
for id_ in REGRESSION_IDS:
    logger.info(f"Running on dataset {id_}...")
    # track scores
    scores = []
    # n_splits cross validation
    for kth in range(n_splits):
        x_train, x_test, y_train, y_test = load_data_by_id(id_, kth, task='regression')
        clf = RandomForestRegressor(n_estimators=500)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        score = r2_score(y_test, y_pred)
        scores.append(score)
        logger.info(f"{kth + 1}-th fold score: {score}.")
    cvs = np.mean(scores)
    logger.info(f"Randomforest on {id_}, {n_splits}-fold cross val score: {cvs}.")
    id_to_score[id_] = cvs
df['randomforest'] = pd.Series(id_to_score)

# lightgbm
id_to_score = {}
for id_ in REGRESSION_IDS:
    logger.info(f"Running on dataset {id_}...")
    # track scores
    scores = []
    # n_splits cross validation
    for kth in range(n_splits):
        x_train, x_test, y_train, y_test = load_data_by_id(id_, kth, task='regression')
        clf = LGBMRegressor()
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        score = r2_score(y_test, y_pred)
        scores.append(score)
        logger.info(f"{kth + 1}-th fold score: {score}.")
    cvs = np.mean(scores)
    logger.info(f"Lightgbm on {id_}, {n_splits}-fold cross val score: {cvs}.")
    id_to_score[id_] = cvs
df['lightgbm'] = pd.Series(id_to_score)

df.to_csv('ensemble_regression.csv')

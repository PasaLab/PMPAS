"""
Data utils used to process dataset.
"""
import os.path as osp

import numpy as np
import pandas as pd

from core.utils.env_util import get_train_test_KFold
from core.utils.log_util import create_logger

logger = create_logger()


def experiment_summary(idx):
    """
    Show the properties of the datasets in the experiments given the id of it.
    :return:
    """
    logger = create_logger()
    logger.info("Dataset name: {}\n")
    logger.info("Total classes of the dataset: {}")
    logger.info("")
    pass


def get_sizeof_dataset(idx, task):
    cur_dir = osp.realpath(__file__)
    data_dir = osp.normpath(
        osp.join(cur_dir, osp.pardir, osp.pardir, osp.pardir, osp.pardir, 'datasets', task,
                 f"{idx}_X.npy"))
    X = np.load(data_dir)
    return round(X.itemsize * X.size / 1048576, 2)

# todo,这个实现不是很优雅，会有循环依赖的问题
def load_data_by_id(id_, k_th=0, task='classification'):
    """
    Split the data, and get the kth fold training and testing data.
    :param id_:
    :param k_th: the k-th fold of the data.
    :param task: classification or regression
    :return:
    """
    cur_dir = osp.realpath(__file__)
    data_dir = osp.normpath(osp.join(cur_dir, osp.pardir, osp.pardir, osp.pardir, osp.pardir, 'datasets', task))
    X = np.load(osp.join(data_dir, f'{id_}_X.npy'), allow_pickle=True)
    y = np.load(osp.join(data_dir, f'{id_}_y.npy'), allow_pickle=True)
    skf = get_train_test_KFold(n_splits=5, task=task)
    train_index, test_index = list(skf.split(X, y))[k_th]
    x_train, x_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    return x_train, x_test, y_train, y_test


def load_data(id_, k_th=0, task='classification'):
    """
    Split the data, and get the kth fold training and testing data.
    :param id_:
    :param k_th: the k-th fold of the data.
    :param task: classification or regression
    :return:
    """
    cur_dir = osp.realpath(__file__)
    data_dir = osp.normpath(osp.join(cur_dir, osp.pardir, osp.pardir, osp.pardir, osp.pardir, 'datasets', task))
    X = np.load(osp.join(data_dir, f'{id_}_X.npy'), allow_pickle=True)
    y = np.load(osp.join(data_dir, f'{id_}_y.npy'), allow_pickle=True)
    skf = get_train_test_KFold(n_splits=5, task=task)
    train_index, test_index = list(skf.split(X, y))[k_th]
    x_train, x_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    return x_train, x_test, y_train, y_test


def get_name_by_id(id_, task='classification'):
    path = rf'core/datasets/{task}/description.csv'
    cur_dir = osp.realpath(__file__)
    data_dir = osp.normpath(osp.join(cur_dir, osp.pardir, osp.pardir, osp.pardir, path))
    df = pd.read_csv(data_dir, index_col=0)
    return df.loc[id_]['name']

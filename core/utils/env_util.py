import argparse
import os
import platform
import random
import warnings

import numpy as np
import psutil
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

from core.utils.constants import ESTIMATORS, REGRESSORS
from core.utils.log_util import create_logger

logger = create_logger()


def environment_init():
    """
    Set running environment.
    :return:
    """
    # todo ,validate seed valid
    random.seed(3)
    np.random.seed(26)
    warnings.filterwarnings("ignore")
    tf.enable_eager_execution()
    tf.set_random_seed(26)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ["PYTHONWARNINGS"] = "ignore"

    # ignore
    def warn(*args, **kwargs):
        pass
    warnings.warn = warn


def get_train_test_KFold(n_splits, task='classification'):
    """
    Using K-fold method to split all the data X into training data and testing data.
    :param n_splits:
    :param task:
    :return:
    """
    if task == 'classification':
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=26)
    elif task == 'regression':
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=26)
    return kf


def split_validation_data(X, y):
    """
    Split the training data into growing data and estimating data.
    :param X:
    :param y:
    :return:
    """
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=26)
    return x_train, x_test, y_train, y_test


def get_adaboost_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', help='Dataset used to be training and testing.', type=int, default=11)
    parser.add_argument('--n_splits', '-n', help='validation n_splits.', type=int, default=5)
    args = parser.parse_args()
    return args


def get_randomforest_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', help='Dataset used to be training and testing.', type=int, default=11)
    parser.add_argument('--n_splits', '-n', help='validation n_splits.', type=int, default=5)
    args = parser.parse_args()
    return args


def get_lightgbm_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', help='Dataset used to be training and testing.', type=int, default=11)
    parser.add_argument('--n_splits', '-n', help='validation n_splits.', type=int, default=5)
    args = parser.parse_args()
    return args


def get_MLP_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_layer_sizes', help='Hidden layer structure of the MLP.', type=tuple,
                        default=(64, 32))
    parser.add_argument('--data', '-d', help='Dataset used to be training and testing.', type=int, default=11)
    parser.add_argument('--n_splits', '-n', help='validation n_splits.', type=int, default=5)
    args = parser.parse_args()
    return args


def get_gbdt_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', help='Dataset used to be training and testing.', type=int, default=11)
    parser.add_argument('--n_splits', '-n', help='validation n_splits.', type=int, default=5)
    args = parser.parse_args()
    return args


def get_xgboost_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', help='Dataset used to be training and testing.', type=int, default=11)
    parser.add_argument('--n_splits', '-n', help='validation n_splits.', type=int, default=5)
    args = parser.parse_args()
    return args


# todo 修改脚本的描述性说明, 将脚本的描述更加的规范,例如增加[]表示相应的参数可以省略,参照shell的声明方式
def get_gcf_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', help='Dataset used to be training and testing.', type=int, default=11)
    parser.add_argument('--n_splits', '-n', help='validation n_splits.', type=int, default=5)
    args = parser.parse_args()
    return args


def get_gcf_cs_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', help='Dataset used to be training and testing.', type=int, default=11)
    parser.add_argument('--n_splits', '-n', help='validation n_splits.', type=int, default=5)
    args = parser.parse_args()
    return args


# todo butongde summary
def EA_summary(data, sp, total_time, limit):
    """
    Summary of the configuration of the experiment.
    :return:
    """
    GB = 2 ** 30

    # todo zheer xuyao xiugai miaohsu ,yinwei meilun yunxing shijian gaibianle
    logger.info(f"EA on {data},search space : {sp}, total running time: {total_time}, "
                f"time limit for per model: {limit}.")
    logger.info(f"""{'*' * 20}Experiment environment{'*' * 20}
    platform: {platform.platform()}"
    architecture: {platform.architecture()}
    processor: {platform.processor()}
    physical cores: {psutil.cpu_count(logical=False)}
    virtual cores: {psutil.cpu_count()}
    total memory: {psutil.virtual_memory().total / GB:.2f}G
    available memory: {psutil.virtual_memory().available / GB:.2f}G
    python version: {platform.python_version()}
    process id: {os.getpid()}.\n""")


# todo 好看一点的方式啦,而且单个的运行时间限制没有传入进来, ea的运行没有为单个模型设置运行时间限制
def ea_summary(data, sp, max_iter):
    """
    Summary of the configuration of the experiment.
    :return:
    """
    GB = 2 ** 30

    # todo zheer xuyao xiugai miaohsu ,yinwei meilun yunxing shijian gaibianle
    logger.info(f"EA on {data}, search space : {sp}, max iterations: {max_iter}.")
    logger.info(f"""{'*' * 20}Experiment environment{'*' * 20}
    platform: {platform.platform()}"
    architecture: {platform.architecture()}
    processor: {platform.processor()}
    physical cores: {psutil.cpu_count(logical=False)}
    virtual cores: {psutil.cpu_count()}
    total memory: {psutil.virtual_memory().total / GB:.2f}G
    available memory: {psutil.virtual_memory().available / GB:.2f}G
    python version: {platform.python_version()}
    process id: {os.getpid()}.\n""")


def ea_time_summary(data, sp, total_time):
    """
    Summary of the configuration of the experiment.
    :return:
    """
    GB = 2 ** 30

    # todo zheer xuyao xiugai miaohsu ,yinwei meilun yunxing shijian gaibianle
    logger.info(f"EA on {data}, "
                f"search space: {sp}, "
                f"total time: {total_time}.")
    logger.info(f"""{'*' * 20}Experiment environment{'*' * 20}
    platform: {platform.platform()}"
    architecture: {platform.architecture()}
    processor: {platform.processor()}
    physical cores: {psutil.cpu_count(logical=False)}
    virtual cores: {psutil.cpu_count()}
    total memory: {psutil.virtual_memory().total / GB:.2f}G
    available memory: {psutil.virtual_memory().available / GB:.2f}G
    python version: {platform.python_version()}
    process id: {os.getpid()}.\n""")


# todo，增加每个模型的运行是件限制
def get_ea_args():
    """
    Get args from the console.
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--space', '-s', help='search space.', default='DAG')
    parser.add_argument('--data', '-d', help='Dataset used to be training and testing.', type=int, default=11)
    parser.add_argument('--Kmost', '-K', help='most k models', type=int, default=8)
    parser.add_argument('--max_cell', help='max cell for search space', type=int, default=8)
    parser.add_argument('--max_iter', help='max iterations rounds', type=int, default=2)
    # todo
    parser.add_argument('--model_time_limit', help='Time limit for model.', type=int, default=None)
    parser.add_argument('--cell_time_limit', help='Time limit for cell.', type=int, default=None)
    parser.add_argument('--kth', '-k', help='kth fold of the data', type=int, default=0)
    parser.add_argument('--n_splits', '-n', help='validation n_splits.', type=int, default=5)
    parse_args = parser.parse_args()
    return parse_args


def get_ea_ray_args():
    """
    Get args from the console.
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--space', '-s', help='search space.', default='plain')
    parser.add_argument('--data', '-d', help='Dataset used to be training and testing.', type=int, default=11)
    parser.add_argument('--Kmost', '-K', help='most k models', type=int, default=8)
    parser.add_argument('--max_cell', help='max cell for search space', type=int, default=8)
    parser.add_argument('--max_iter', help='max iterations rounds', type=int, default=10)
    parser.add_argument('--stages', help='parallel stages', type=int, default=2)
    # todo
    parser.add_argument('--model_time_limit', help='Time limit for model.', type=int, default=None)
    parser.add_argument('--cell_time_limit', help='Time limit for cell.', type=int, default=120)
    parser.add_argument('--kth', '-k', help='kth fold of the data', type=int, default=0)
    parser.add_argument('--n_splits', '-n', help='validation n_splits.', type=int, default=5)
    parse_args = parser.parse_args()
    return parse_args


def get_ea_reg_args():
    """
    Get args from the console.
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--space', '-s', help='search space.', default='DAG')
    parser.add_argument('--data', '-d', help='Dataset used to be training and testing.', type=int, default=201)
    parser.add_argument('--max_iter', help='max iterations rounds', type=int, default=100)
    # todo
    parser.add_argument('--model_time_limit', help='Time limit for model.', type=int, default=None)
    parser.add_argument('--cell_time_limit', help='Time limit for cell.', type=int, default=120)
    parser.add_argument('--kth', '-k', help='kth fold of the data', type=int, default=0)
    parser.add_argument('--n_splits', '-n', help='validation n_splits.', type=int, default=5)
    parse_args = parser.parse_args()
    return parse_args


def get_ea_time_args():
    """
    Get args from the console.
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--space', '-s', help='search space.', default='DAG')
    parser.add_argument('--data', '-d', help='Dataset used to be training and testing.', type=int, default=11)
    parser.add_argument('--Kmost', '-K', help='most k models', type=int, default=8)
    parser.add_argument('--max_iter', help='max iterations rounds', type=int, default=100000)
    parser.add_argument('--total_time', '-t', help='Total time for evolutionary evolution_controller', type=int, default=3600 * 10)
    # todo
    parser.add_argument('--model_time_limit', help='Time limit for model.', type=int, default=None)
    parser.add_argument('--cell_time_limit', help='Time limit for cell.', type=int, default=120)
    parser.add_argument('--kth', '-k', help='kth fold of the data', type=int, default=0)
    parser.add_argument('--n_splits', '-n', help='validation n_splits.', type=int, default=5)
    parse_args = parser.parse_args()
    return parse_args


def get_bdas_time_args():
    """
    Get args from the console.
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', help='Dataset used to be training and testing.', type=int, default=11)
    parser.add_argument('--space', '-s', help='search space.', default='DAG')
    parser.add_argument('--Kmost', '-K', help='most k models.', type=int, default=8)
    parser.add_argument('--max_cell', '-C', help='Maximum cell number of a layer.', type=int, default=1000)
    parser.add_argument('--total_time', '-t', help='Total time budget for the evolution_controller.', type=int, default=30)
    # todo
    parser.add_argument('--cell_time_limit', help='Time limit for cell.', type=int, default=120)
    parser.add_argument('--model_time_limit', help='Time limit for model.', type=int, default=None)
    parser.add_argument('--estimators', help='Possible estimators for cell.', nargs='+', default=ESTIMATORS)
    parser.add_argument('--exclude_estimators', help='Excluding some estimator for cell.', nargs='+',
                        default=None)
    parser.add_argument('--kth', '-k', help='kth fold of the data', type=int, default=0)
    parser.add_argument('--strategy', help='strategy used by the controller', type=str, default='best')
    parse_args = parser.parse_args()
    return parse_args


def get_bdas_time_cv_args():
    """
    Get args from the console.
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', help='Dataset used to be training and testing.', type=int, default=11)
    parser.add_argument('--space', '-s', help='search space.', default='DAG')
    parser.add_argument('--Kmost', '-K', help='most k models.', type=int, default=8)
    parser.add_argument('--max_cell', '-C', help='Maximum cell number of a layer.', type=int, default=1000)
    parser.add_argument('--total_time', '-t', help='Total time budget for the evolution_controller.', type=int, default=30)
    # todo
    parser.add_argument('--cell_time_limit', help='Time limit for cell.', type=int, default=120)
    parser.add_argument('--model_time_limit', help='Time limit for model.', type=int, default=None)
    parser.add_argument('--estimators', help='Possible estimators for cell.', nargs='+', default=ESTIMATORS)
    parser.add_argument('--exclude_estimators', help='Excluding some estimator for cell.', nargs='+',
                        default=None)
    parser.add_argument('--kth', '-k', help='kth fold of the data', type=int, default=0)
    parser.add_argument('--n_splits', '-n', help='validation n_splits.', type=int, default=5)
    parser.add_argument('--strategy', help='strategy used by the controller', type=str, default='best')
    parse_args = parser.parse_args()
    return parse_args


def get_bdas_reg_time_args():
    """
    Get args from the console.
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', help='Dataset used to be training and testing.', type=int, default=201)
    parser.add_argument('--space', '-s', help='search space.', default='DAG')
    parser.add_argument('--Kmost', '-K', help='most k models.', type=int, default=8)
    parser.add_argument('--max_cell', '-C', help='Maximum cell number of a layer.', type=int, default=1000)
    parser.add_argument('--total_time', '-t', help='Total time budget for the evolution_controller.', type=int, default=30)
    # todo
    parser.add_argument('--cell_time_limit', help='Time limit for cell.', type=int, default=120)
    parser.add_argument('--model_time_limit', help='Time limit for model.', type=int, default=None)
    parser.add_argument('--estimators', help='Possible estimators for cell.', nargs='+', default=REGRESSORS)
    parser.add_argument('--exclude_estimators', help='Excluding some estimator for cell.', nargs='+',
                        default=None)
    parser.add_argument('--kth', '-k', help='kth fold of the data', type=int, default=0)
    parser.add_argument('--n_splits', '-n', help='validation n_splits.', type=int, default=5)
    parser.add_argument('--strategy', help='strategy used by the controller', type=str, default='best')
    parse_args = parser.parse_args()
    return parse_args


def get_bdas_reg_time_cv_args():
    """
    Get args from the console.
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', help='Dataset used to be training and testing.', type=int, default=201)
    parser.add_argument('--space', '-s', help='search space.', default='DAG')
    parser.add_argument('--Kmost', '-K', help='most k models.', type=int, default=8)
    parser.add_argument('--max_cell', '-C', help='Maximum cell number of a layer.', type=int, default=1000)
    parser.add_argument('--total_time', '-t', help='Total time budget for the evolution_controller.', type=int, default=3600)
    # todo
    parser.add_argument('--cell_time_limit', help='Time limit for cell.', type=int, default=120)
    parser.add_argument('--model_time_limit', help='Time limit for model.', type=int, default=None)
    parser.add_argument('--estimators', help='Possible estimators for cell.', nargs='+', default=REGRESSORS)
    parser.add_argument('--exclude_estimators', help='Excluding some estimator for cell.', nargs='+',
                        default=None)
    parser.add_argument('--kth', '-k', help='kth fold of the data', type=int, default=0)
    parser.add_argument('--n_splits', '-n', help='validation n_splits.', type=int, default=5)
    parser.add_argument('--strategy', help='strategy used by the controller', type=str, default='best')
    parse_args = parser.parse_args()
    return parse_args


def get_bdas_cell_args():
    """
    Get args from the console.
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', help='Dataset used to be training and testing.', type=int, default=11)
    parser.add_argument('--space', '-s', help='search space.', default='DAG')
    parser.add_argument('--Kmost', '-K', help='most k models.', type=int, default=8)
    parser.add_argument('--max_cell', '-C', help='Maximum cell number of a layer.', type=int, default=3)
    # todo
    parser.add_argument('--cell_time_limit', help='Time limit for cell.', type=int, default=120)
    parser.add_argument('--model_time_limit', help='Time limit for model.', type=int, default=None)
    parser.add_argument('--estimators', help='Possible estimators for cell.', nargs='+', default=ESTIMATORS)
    parser.add_argument('--exclude_estimators', help='Excluding some estimator for cell.', nargs='+',
                        default=None)
    parser.add_argument('--kth', '-k', help='kth fold of the data', type=int, default=0)
    parser.add_argument('--strategy', help='strategy used by the controller', type=str, default='best')
    parse_args = parser.parse_args()
    return parse_args


def get_autosklearn_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', help='Dataset used to be training and testing.', type=int, default=201)
    parser.add_argument('--total_time', '-t', help='Total time budget for the evolution_controller.', type=int, default=3600)
    parser.add_argument('--n_splits', '-n', help='validation n_splits.', type=int, default=5)
    parse_args = parser.parse_args()
    return parse_args

def get_autosklearn_cls_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', help='Dataset used to be training and testing.', type=int, default=6)
    parser.add_argument('--total_time', '-t', help='Total time budget for the evolution_controller.', type=int, default=3600)
    parser.add_argument('--n_splits', '-n', help='validation n_splits.', type=int, default=5)
    parse_args = parser.parse_args()
    return parse_args


def get_h2o_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', help='Dataset used to be training and testing.', type=int, default=201)
    parser.add_argument('--total_time', '-t', help='Total time budget for the evolution_controller.', type=int, default=3600)
    parser.add_argument('--n_splits', '-n', help='validation n_splits.', type=int, default=5)
    parse_args = parser.parse_args()
    return parse_args


def get_bdas_cell_args():
    """
    Get args from the console.
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', help='Dataset used to be training and testing.', type=int, default=11)
    parser.add_argument('--space', '-s', help='search space.', default='DAG')
    parser.add_argument('--Kmost', '-K', help='most k models.', type=int, default=8)
    parser.add_argument('--max_cell', '-C', help='Maximum cell number of a layer.', type=int, default=8)
    # todo
    parser.add_argument('--cell_time_limit', help='Time limit for cell.', type=int, default=120)
    parser.add_argument('--model_time_limit', help='Time limit for model.', type=int, default=120 * 8)
    parser.add_argument('--estimators', help='Possible estimators for cell.', nargs='+', default=ESTIMATORS)
    parser.add_argument('--exclude_estimators', help='Excluding some estimator for cell.', nargs='+',
                        default=None)
    parser.add_argument('--kth', '-k', help='kth fold of the data', type=int, default=0)
    parser.add_argument('--strategy', help='strategy used by the controller', type=str, default='best')
    parse_args = parser.parse_args()
    return parse_args


def get_bdas_cell_ray_args():
    """
    Get args from the console.
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', help='Dataset used to be training and testing.', type=int, default=11)
    parser.add_argument('--space', '-s', help='search space.', default='DAG')
    parser.add_argument('--Kmost', '-K', help='most k models.', type=int, default=8)
    parser.add_argument('--stages', help='parallel level', type=int, default=2)
    parser.add_argument('--max_cell', '-C', help='Maximum cell number of a layer.', type=int, default=8)
    # todo
    parser.add_argument('--cell_time_limit', help='Time limit for cell.', type=int, default=120)
    parser.add_argument('--model_time_limit', help='Time limit for model.', type=int, default=120 * 8)
    parser.add_argument('--estimators', help='Possible estimators for cell.', nargs='+', default=ESTIMATORS)
    parser.add_argument('--exclude_estimators', help='Excluding some estimator for cell.', nargs='+',
                        default=None)
    parser.add_argument('--kth', '-k', help='kth fold of the data', type=int, default=0)
    parser.add_argument('--strategy', help='strategy used by the controller', type=str, default='best')
    parse_args = parser.parse_args()
    return parse_args


def get_bdas_cell_cv_args():
    """
    Get args from the console.
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', help='Dataset used to be training and testing.', type=int, default=11)
    parser.add_argument('--space', '-s', help='search space', default='DAG')
    parser.add_argument('--Kmost', '-K', help='most k models', type=int, default=8)
    parser.add_argument('--max_cell', '-C', help='Maximum cell number of a layer', type=int, default=8)
    parser.add_argument('--n_splits', '-n', help='validation n_splits.', type=int, default=5)
    parser.add_argument('--cell_time_limit', help='Time limit for cell.', type=int, default=120)
    parser.add_argument('--model_time_limit', help='Time limit for model.', type=int, default=None)
    parser.add_argument('--estimators', help='Possible estimators for cell.', nargs='+', default=ESTIMATORS)
    parser.add_argument('--exclude_estimators', help='Excluding some estimator for cell.', nargs='+',
                        default=None)
    parser.add_argument('--strategy', help='strategy used by the controller', type=str, default='best')
    parse_args = parser.parse_args()
    return parse_args


# todo，增加每个模型的运行时间限制
def get_bdas_B_args():
    """
    Get args from the console.
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', help='Dataset used to be training and testing.', type=int, default=11)
    parser.add_argument('--space', '-s', help='search space', default='DAG')
    parser.add_argument('--Kmost', '-K', help='most k models', type=int, default=8)
    parser.add_argument('--block', '-B', help='Maximum block number', type=int, default=8)
    # todo
    parser.add_argument('--limit', '-l', help='time limit for per model', type=int, default=120)
    parse_args = parser.parse_args()
    return parse_args


def get_ensemble_args():
    """"
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', help='Dataset used to be training and testing.', type=int, default=11)
    parse_args = parser.parse_args()
    return parse_args


def get_stable_args():
    """
    Get args from the console of the stable search process.
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--total_time', '-t', help='Total running time of the process.', type=int, default=30)
    parser.add_argument('--data', '-d', help='Dataset used to be training and testing.', type=int, default=11)
    parser.add_argument('--space', '-s', help='search space', default='DAG')
    # todo
    parser.add_argument('--limit', '-l', help='time limit for per model', type=int, default=120)
    parse_args = parser.parse_args()
    # todo,增加可以改变变异选项的参数,即输入的是一个列表
    return parse_args


def get_random_search_args():
    """
    Get args from the console of the random search process.
    :return: parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', help='Dataset used to be training and testing.', type=int, default=11)
    parser.add_argument('--space', '-s', help='search space.', default='DAG')
    parser.add_argument('--max_cell', '-C', help='Maximum cell number of a layer.', type=int, default=8)
    parser.add_argument('--total_time', '-t', help='Total running time of the process.', type=int, default=360)
    parser.add_argument('--cell_time_limit', help='Time limit for cell.', type=int, default=120)
    parser.add_argument('--model_time_limit', help='Time limit for model.', type=int, default=None)
    parser.add_argument('--estimators', help='Possible estimators for cell.', nargs='+', default=ESTIMATORS)
    parser.add_argument('--exclude_estimators', help='Excluding some estimator for cell.', nargs='+',
                        default=None)
    parser.add_argument('--kth', '-k', help='kth fold of the data', type=int, default=0)
    parse_args = parser.parse_args()
    return parse_args

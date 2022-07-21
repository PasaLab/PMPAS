import os
import random
import warnings

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, r2_score

from core.search_algorithm import EDEAS, PDEAS
from core.utils.log_util import create_logger

logger = create_logger()


class SearchSpace:
    def __init__(self, space_type):
        self.space_type = None
        if space_type == 'CP':
            self.space_type = 'CP'
        elif space_type == 'DAG':
            self.space_type = 'DAG'
        else:
            raise NotImplementedError("Not supported search space.")
        self.name = self.space_type


class SearchAlgorithm:
    def __init__(self, method, task='classification', K=8, strategy='best', max_cell=1000, cell_time_limit=120,
                 model_time_limit=None, encOperator='BG', NIND=8, data_dir=None, confidence_screen=False):
        self.impl = None

        if task == 'regression' and confidence_screen:
            raise ValueError("Regression task doesn't support Confidence Screening mechanism.")

        if method == 'EPEAAS':
            self.impl = EDEAS(encOperator=encOperator, NIND=NIND, max_cell=max_cell, data_dir=data_dir, task=task,
                              confidence_screen=confidence_screen)
        elif method == 'PMPAS':
            self.impl = PDEAS(K=K, task=task, model_time_limit=model_time_limit, cell_time_limit=cell_time_limit,
                              max_cell=max_cell, strategy=strategy, data_dir=data_dir,
                              confidence_screen=confidence_screen)
        else:
            raise NotImplementedError("Not supported search method.")

        self.name = method

    def get_name(self):
        """
        Get string representation of the search algorithm.
        :return:
        """
        return self.name

    pass


class AutoDEL:
    def __init__(self, search_space, search_algorithm, task, total_budget, random_state=26, budget_type='running_time',
                 is_parallel=False):
        """
        Core class of automated deep ensemble learning system.
        :param search_space: Search Space
        :param search_algorithm: Search Algorithm
        :param task: Which task to perform, optionally ['classification','regression']
        :param total_budget: Total budget for the search process, currently supporting total running time budget, for example, 3600s
        :param random_state: Random state to controlling the whole search process.
        :param budget_type: Fixed, currently supporting 'runnning_time','max_iter','max_cell'.Running time and maximum
        iterations for EPEAAS and running time and maximum cell for PMPAS.
        :param is_parallel: Whether to perform ray to speed up the whole search process.
        """
        self.search_space = search_space
        self.search_algorithm = search_algorithm
        self.task = task
        if self.task != self.search_algorithm.impl.task:
            raise ValueError("Task passed to search algorithm and AutoDEL must be identity.")
        self.budget_type = budget_type
        if self.budget_type not in ['running_time', 'max_iter', 'max_cell']:
            raise ValueError("Not supported budget type.")
        self.total_budget = total_budget
        try:
            self.total_budget = int(self.total_budget)
        except:
            raise TypeError("Not supported budget, must passing string or int.")
        self.random_state = random_state
        self.is_parallel = is_parallel

        self._set_random_state()
        self._set_search_algorithm()

        # final result
        self.final_score = None

    def _get_budget_type_name(self):
        if self.budget_type == 'running_time':
            budget_type_name = "maximum running time"
        elif self.budget_type == 'max_iter':
            budget_type_name = 'maximum iterations'
        elif self.budget_type == 'max_cell':
            budget_type_name = 'maximum Cell'
        else:
            raise ValueError("Not supported budget_type... ")
        return budget_type_name

    def experiment_summary(self):
        """
        Experiment summary of the whole search process.
        :return:
        """
        logger.info(f"AutoDEL experiment summary...")
        logger.info(f"Search space: {self.search_space.name}.")
        logger.info(f"Search algorithm: {self.search_algorithm.name}.")
        logger.info(f"Budget type: {self._get_budget_type_name()}, total budget: {self.total_budget}.")
        logger.info(
            f"Using Ray: {self.is_parallel}, "
            f"using confidence screen: {self.search_algorithm.impl.confidence_screen}.")

        # logging args
        args = self.search_algorithm.impl._get_algorithm_hyperparameters()
        logger.info(f"Algorithm hyperparameters: {args}.\n")

    def search_with_algorithm(self, x_train, y_train):
        self.search_algorithm.impl.run(x_train, y_train)
        pass

    def refit_and_predict(self, x_train, y_train, x_test):
        logger.info("AutoDEL search process finished, begin to refit and score...")
        return self.search_algorithm.impl.refit_and_predict(x_train, y_train, x_test)

    def score(self, y_pred, y_test):
        """
        Score function for the final predictions. Optionally 'accuracy' or 'acc' for classification task,
        'r2' for regression task.
        :param y_pred:
        :param y_test:
        :return:
        """
        if self.task == 'classification':
            self.final_score = accuracy_score(y_test, y_pred)
        elif self.task == 'regression':
            self.final_score = r2_score(y_test, y_pred)
        else:
            raise ValueError("Not supported scoring function.")
        return self.final_score

    def result_summary(self):
        """
        Summary of the result.
        :return:
        """
        logger.info(f"Final score: {self.final_score}.")

    def _set_search_algorithm(self):
        # set task
        self.search_algorithm.impl.task = self.task

        # set parallel
        self.search_algorithm.impl.is_parallel = self.is_parallel

        # set budget
        if self.budget_type == 'running_time':
            self.search_algorithm.impl.stop_by_time = True
            self.search_algorithm.impl.total_time = self.total_budget
        elif self.budget_type == 'max_iter':
            self.search_algorithm.impl.stop_by_time = False
            self.search_algorithm.impl.max_iter = self.total_budget
        elif self.budget_type == 'max_cell':
            self.search_algorithm.impl.stop_by_time = False
            self.search_algorithm.impl.max_cell = self.total_budget
        else:
            raise ValueError("Not supported budget type.")

        # set search space string
        self.search_algorithm.impl.space = self.search_space.space_type

        # set search space
        self.search_algorithm.impl._set_search_space()

    def _set_random_state(self):
        # set random seed for random, numpy and tensorflow
        # ignore warnings
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        warnings.filterwarnings("ignore")
        tf.enable_eager_execution()
        tf.set_random_seed(self.random_state)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.environ["PYTHONWARNINGS"] = "ignore"

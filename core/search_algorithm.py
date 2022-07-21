import csv
import os
import os.path as osp
import time
import numpy as np

import psutil
import ray

import geatpy as ea

from core.edeas.evolution_controller.SEGA_v2 import soea_SEGA_templet
from core.edeas.search_space.cp_space import CPSpace as EACPSpace
from core.edeas.search_space.dag_space import DAGSpace as EADAGSpace
from core.pdeas.model_manager.model_manager_classification import Model as Model_cls
from core.pdeas.model_manager.model_manger_regression import Model as Model_reg
from core.pdeas.proxy_model import ProxyModel
from core.pdeas.search_space import CPSpace as PCPSpace, Evaluator
from core.pdeas.search_space import DAGSpace as PDAGSpace
from core.utils.constants import ESTIMATORS, REGRESSORS
from core.utils.env_util import split_validation_data
from core.utils.log_util import create_logger

# get the logger to logging experiments
logger = create_logger()


class SearchAlgorithm:
    def __init__(self):
        pass

    def run(self, x_train, y_train):
        pass

    def refit_and_predict(self, x_train, y_train, x_test):
        pass

    def _get_algorithm_hyperparameters(self):
        """
        Return hyperparameters of the algoirhtm.
        :return:
        """
        pass

    pass


class EDEAS(SearchAlgorithm):
    def __init__(self, selOperator='tour', encOperator='BG', recOperator=None, mutOperator=None, evaluator='direct',
                 NIND=8, max_cell=8, data_dir=None,
                 task='classification', is_parallel=False, confidence_screen=False):
        """

        :param selOperator: Seleciton operator.
        :param encOperator: Encoding operator.
        :param recOperator: Recombination operator.
        :param mutOperator: Mutation operator.
        :param evaluator: Evaluator.
        :param NIND: Size of the population.
        :param max_cell: Maximum Cell of the search space.
        :param data_dir: Which path to save the result.
        :param task: Task.
        """
        super().__init__()

        self.selOperator = selOperator
        self.recOperator = recOperator
        self.mutOperator = mutOperator
        self.encOperator = encOperator
        self.evaluator = evaluator

        self.problem = None
        self.Encoding = encOperator
        self.NIND = NIND
        self.max_cell = max_cell
        self.task = task
        self.data_dir = data_dir

        self.BestIndi = None

        # passed data
        self.X_train = None
        self.X_val = None
        self.Y_train = None
        self.Y_val = None

        # time budget
        self.total_time = None

        # max iterations
        self.max_iter = None

        # whether stopping by time
        self.stop_by_time = False

        # space
        self.space = None
        self.problem = None

        # whether parallel
        self.is_parallel = is_parallel

        # whether confidence screen
        self.confidence_screen = confidence_screen

        if self.task == 'classification':
            self.estimators = ESTIMATORS
        elif self.task == 'regression':
            self.estimators = REGRESSORS
        else:
            raise ValueError('Not supported task.')
        self.name = "EPEAAS"

    def _set_search_space(self):
        # create directory to save the result
        time_str = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time()))
        RESULTS_SAVED_PATH = f"edeas_{self.space}_cell_{self.max_cell}_{time_str}_max_iter_{self.max_iter}"
        cur_dir = osp.realpath(__file__)
        self.data_dir = osp.normpath(osp.join(cur_dir, osp.pardir, osp.pardir, 'results', RESULTS_SAVED_PATH))
        if not osp.exists(self.data_dir):
            os.makedirs(self.data_dir)

        # set task
        if self.task == 'classification':
            if self.space == 'DAG' or self.space == 'dag':
                self.problem = EADAGSpace(self.X_train, self.Y_train, self.X_val, self.Y_val,
                                          is_parallel=self.is_parallel, confidence_screen=self.confidence_screen)
            elif self.space == 'plain' or self.space == 'PLAIN' or self.space == 'CP' or self.space == 'cp':
                self.problem = EACPSpace(self.X_train, self.Y_train, self.X_val, self.Y_val,
                                         is_parallel=self.is_parallel, confidence_screen=self.confidence_screen)
            else:
                raise ValueError("Not supported search space.")
        elif self.task == 'regression':
            if self.space == 'DAG' or self.space == 'dag':
                self.problem = EADAGSpace(self.X_train, self.Y_train, self.X_val, self.Y_val, task='regression',
                                          is_parallel=self.is_parallel, confidence_screen=self.confidence_screen)
            elif self.space == 'plain' or self.space == 'PLAIN' or self.space == 'CP' or self.space == 'cp':
                self.problem = EACPSpace(self.X_train, self.Y_train, self.X_val, self.Y_val, task='regression',
                                         is_parallel=self.is_parallel, confidence_screen=self.confidence_screen)
            else:
                raise ValueError("Not supported search space.")
        else:
            raise ValueError("Not supported task.")

    def _set_search_space_dataset(self, x_train, y_train):
        X_train, X_val, Y_train, Y_val = split_validation_data(x_train, y_train)
        self.problem.X_train = X_train
        self.problem.Y_train = Y_train
        self.problem.X_val = X_val
        self.problem.Y_val = Y_val

    def _get_algorithm_hyperparameters(self):
        args = {
            "population size": self.NIND,
            "selection operator": self.selOperator,
            "encoding operator": self.encOperator,
        }
        return args

    def run(self, x_train, y_train):
        # set dataset for the search space
        self._set_search_space_dataset(x_train, y_train)

        # track the best model and it's score
        self.best_model = None
        self.best_score = 0

        # running exit flag and timer, used to break the loop and track the running time
        self.timeout_flag = False
        self.start_time = time.time()
        """=================================种群设置=============================="""

        # population settings
        Field = ea.crtfld(self.Encoding, self.problem.varTypes, self.problem.ranges,
                          self.problem.borders)  # 创建区域描述器
        population = ea.Population(self.Encoding, Field, self.NIND)  # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）

        # evolution_controller settings
        """===============================算法参数设置============================="""
        myAlgorithm = soea_SEGA_templet(self.problem, population, self.selOperator, self.recOperator, self.mutOperator,
                                        self.encOperator, result_path=self.data_dir,
                                        stop_by_time=self.stop_by_time)  # 实例化一个算法模板对象

        if self.stop_by_time:
            myAlgorithm.MAXTIME = self.total_time
        else:
            myAlgorithm.MAXGEN = self.max_iter

        myAlgorithm.logTras = 1  # 设置每隔多少代记录日志，若设置成0则表示不记录日志
        myAlgorithm.verbose = False  # 设置是否打印输出日志信息
        myAlgorithm.drawing = 0  # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画；3：绘制决策空间过程动画）
        """==========================调用算法模板进行种群进化========================"""

        # run evolution_controller template
        [self.BestIndi, self.population] = myAlgorithm.run()  # 执行算法模板，得到最优个体以及最后一代种群

    def refit_and_predict(self, x_train, y_train, x_test):
        if self.BestIndi.sizes != 0:
            # get best model and get its score
            best_index = self.BestIndi.Phen.flatten()
            best_model = self.problem.best_model
            best_model.refit(x_train, y_train)
            y_pred = best_model.predict(x_test)
            return y_pred


class PDEAS(SearchAlgorithm):
    def __init__(self, K=8, task='classification', model_time_limit=None,
                 cell_time_limit=120, max_cell=1000,
                 strategy='best', data_dir=None, is_parallel=False, confidence_screen=False):
        super().__init__()

        self.K = K
        self.model_time_limit = model_time_limit
        self.cell_time_limit = cell_time_limit

        self.max_cell = max_cell
        self.strategy = strategy
        self.data_dir = data_dir
        self.task = task

        # passed data
        self.X_train = None
        self.X_val = None
        self.Y_train = None
        self.Y_val = None

        # whether stopping by time
        self.stop_by_time = False

        # time budget
        self.total_time = None

        # ray distribute
        self.is_parallel = is_parallel

        # whether using confidence screen
        self.confidence_screen = confidence_screen

        # data id
        self.data = None

        # space
        self.space = None

        if self.task == 'classification':
            self.estimators = ESTIMATORS
        elif self.task == 'regression':
            self.estimators = REGRESSORS
        else:
            raise ValueError('Not supported task.')
        self.name = "PMPAS"

    def _set_search_space(self):
        # create directory to save the result
        time_str = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time()))
        RESULTS_SAVED_PATH = f"pdeas_{time_str}_{self.space}_{self.K}_total_time_{self.total_time}_data_{self.data}"
        cur_dir = osp.realpath(__file__)
        self.data_dir = osp.normpath(osp.join(cur_dir, osp.pardir, osp.pardir, 'results', RESULTS_SAVED_PATH))
        if not osp.exists(self.data_dir):
            os.makedirs(self.data_dir)

        # initializing search space and controller
        if self.space == 'DAG' or self.space == 'dag':
            sp = PDAGSpace(self.max_cell, self.estimators)
        elif self.space == 'CP' or self.space == 'cp' or self.space == 'plain' or self.space == 'PLAIN':
            sp = PCPSpace(self.max_cell, self.estimators)
        elif self.space is None:
            raise ValueError("Must specified search space.")
        else:
            raise ValueError("Not supported search space.")
        self.controller = ProxyModel(sp, self.K, saved_path=self.data_dir, strategy=self.strategy)

    def _get_algorithm_hyperparameters(self):
        args = {
            'K': self.K,
            'preserved strategy': self.strategy,
            'cell time limit': self.cell_time_limit,
            'model time limit': self.model_time_limit,
        }
        return args

    def _run_with_time_budget(self, x_train, y_train):
        # load dataset
        self.X_train, self.X_val, self.Y_train, self.Y_val = split_validation_data(x_train, y_train)

        # running exit flag and timer, used to break the loop and track the running time
        self.timeout_flag = False
        self.start_time = time.time()

        # record average score and best score
        self.cell_average_scores = []
        self.cell_best_scores = []

        # track the best model and it's score
        self.best_model = None
        self.best_score = 0
        self.model_count = 0
        self.trial = 0

        # loop until reach time budget
        while True:
            # begin the growing process
            if self.trial == 0:
                k = None
            else:
                k = self.K
            logger.info(f"Begin cell number = {self.trial + 1}")
            # get all the model for the current trial
            actions = self.controller.get_models(top_k=k)
            rewards = []
            for t, action in enumerate(actions, 1):
                logger.info(f"Current #{t} / #{len(actions)}.")
                logger.info(action)
                if self.task == 'classification':
                    cc = Model_cls(cas_config=action, model_time_limit=self.model_time_limit,
                                   cell_time_limit=self.cell_time_limit)
                elif self.task == 'regression':
                    cc = Model_reg(cas_config=action, model_time_limit=self.model_time_limit,
                                   cell_time_limit=self.cell_time_limit)
                else:
                    raise ValueError("Not supported task.")
                try:
                    cc.fit(self.X_train, self.Y_train, self.X_val, self.Y_val)
                except TimeoutError:
                    logger.info("Time limit for the current model has reached.")
                    logger.info(f"#{self.model_count}, score: 0.\n")
                    self.model_count += 1
                    rewards.append(0)
                    continue
                # todo 这儿修改为except Exception 似乎更好
                # todo 运行失败的时候设置score为0比较好
                # todo 修改保存的文件
                except:
                    logger.info(f"{action} running failed on dataset: {self.data}.")
                    logger.info(f"#{self.model_count}, score: 0.\n")
                    self.model_count += 1
                    rewards.append(0)
                    continue
                self.model_count += 1
                reward = cc.cas_model.best_score
                # record the best model
                if reward > self.best_score:
                    self.best_model = cc
                    self.best_score = reward
                logger.info(f"#{self.model_count}, score: {reward}.\n")
                rewards.append(reward)
                # write the results of this trial into a file
                # todo 修改保存的方法
                with open(osp.join(self.data_dir, 'train_history.csv'), mode='a+', newline='') as f:
                    current_data = [reward]
                    current_data.extend(action)
                    writer = csv.writer(f)
                    writer.writerow(current_data)
                # todo 更好的方式来修改运行时间限制
                if time.time() - self.start_time > self.total_time:
                    self.timeout_flag = True
                    break
            if time.time() - self.start_time > self.total_time:
                self.timeout_flag = True
            if self.timeout_flag:
                logger.info("Total time budget has reached.")
                break
            # train and update controller
            loss = self.controller.finetune_proxy_model(rewards)
            self.controller.update_search_space()
            self.trial += 1

    def _run_with_max_cell_budget(self, x_train, y_train):
        # load dataset
        self.X_train, self.X_val, self.Y_train, self.Y_val = split_validation_data(x_train, y_train)

        # track the best model and it's score
        self.best_model = None
        self.best_score = 0
        self.model_count = 0
        self.trial = 0

        # loop until reach time budget
        for trial in range(self.max_cell):
            # begin the growing process
            if self.trial == 0:
                k = None
            else:
                k = self.K
            logger.info(f"Begin cell number = {self.trial + 1}")
            # get all the model for the current trial
            actions = self.controller.get_models(top_k=k)
            rewards = []
            for t, action in enumerate(actions, 1):
                logger.info(f"Current #{t} / #{len(actions)}.")
                logger.info(action)
                if self.task == 'classification':
                    cc = Model_cls(cas_config=action, model_time_limit=self.model_time_limit,
                                   cell_time_limit=self.cell_time_limit)
                elif self.task == 'regression':
                    cc = Model_reg(cas_config=action, model_time_limit=self.model_time_limit,
                                   cell_time_limit=self.cell_time_limit)
                else:
                    raise ValueError("Not supported task.")
                try:
                    cc.fit(self.X_train, self.Y_train, self.X_val, self.Y_val)
                except TimeoutError:
                    logger.info("Time limit for the current model has reached.")
                    logger.info(f"#{self.model_count}, score: 0.\n")
                    self.model_count += 1
                    rewards.append(0)
                    continue
                except:
                    logger.info(f"{action} running failed on dataset: {self.data}.")
                    logger.info(f"#{self.model_count}, score: 0.\n")
                    self.model_count += 1
                    rewards.append(0)
                    continue
                self.model_count += 1
                reward = cc.cas_model.best_score
                # record the best model
                if reward > self.best_score:
                    self.best_model = cc
                    self.best_score = reward
                logger.info(f"#{self.model_count}, score: {reward}.\n")
                rewards.append(reward)
            # train and update controller
            loss = self.controller.finetune_proxy_model(rewards)
            self.controller.update_search_space()
            self.trial += 1

    def _run_with_time_budget_ray(self, x_train, y_train):
        # load dataset
        self.X_train, self.X_val, self.Y_train, self.Y_val = split_validation_data(x_train, y_train)

        # running exit flag and timer, used to break the loop and track the running time
        self.timeout_flag = False
        self.start_time = time.time()

        # track the best model and it's score
        self.best_model = None
        self.best_score = 0
        self.model_count = 0
        self.trial = 0

        # begin the growing process
        all_models = []
        all_scores = []
        while True:
            for trial in range(self.max_cell):
                if trial == 0:
                    k = None
                else:
                    k = self.K
                logger.info(f"Begin cell number = {trial + 1}")

                # get all the model for the current trial
                actions = self.controller.get_models(top_k=k)

                evaluators = [Evaluator.remote(action, model_time_limit=self.model_time_limit,
                                               cell_time_limit=self.cell_time_limit, task=self.task) for action in
                              actions]
                [evaluator.evaluate.remote(self.X_train, self.Y_train, self.X_val, self.Y_val) for evaluator in
                 evaluators]

                scores = ray.get([evaluator.get_score.remote() for evaluator in evaluators])
                models = ray.get([evaluator.get_model.remote() for evaluator in evaluators])

                all_scores.extend(scores)
                all_models.extend(models)
                if time.time() - self.start_time > self.total_time:
                    self.timeout_flag = True
                    break
                # train and update controller
                loss = self.controller.finetune_proxy_model(scores)
                self.controller.update_search_space()
                trial += 1
            if time.time() - self.start_time > self.total_time:
                self.timeout_flag = True
            if self.timeout_flag:
                logger.info("Total time budget has reached.")
                break

        index = np.argmax(all_scores)
        self.best_model = all_models[index]
        logger.info("Maximum cell has reached.\n")

    def _run_with_max_cell_budget_ray(self, x_train, y_train):
        # load dataset
        self.X_train, self.X_val, self.Y_train, self.Y_val = split_validation_data(x_train, y_train)

        # track the best model and it's score
        self.best_model = None
        self.best_score = 0
        self.model_count = 0
        self.trial = 0

        # begin the growing process
        all_models = []
        all_scores = []
        for trial in range(self.max_cell):
            if trial == 0:
                k = None
            else:
                k = self.K
            logger.info(f"Begin cell number = {trial + 1}")

            # get all the model for the current trial
            actions = self.controller.get_models(top_k=k)

            evaluators = [Evaluator.remote(action, model_time_limit=self.model_time_limit,
                                           cell_time_limit=self.cell_time_limit, task=self.task) for action in actions]
            [evaluator.evaluate.remote(self.X_train, self.Y_train, self.X_val, self.Y_val) for evaluator in evaluators]

            scores = ray.get([evaluator.get_score.remote() for evaluator in evaluators])
            models = ray.get([evaluator.get_model.remote() for evaluator in evaluators])

            all_scores.extend(scores)
            all_models.extend(models)

            # train and update controller
            loss = self.controller.finetune_proxy_model(scores)
            self.controller.update_search_space()
            trial += 1

        index = np.argmax(all_scores)
        self.best_model = all_models[index]
        logger.info("Maximum cell has reached.\n")

    def run(self, x_train, y_train):
        """
        :param x_train:
        :param y_train:
        :return:
        """
        if self.is_parallel:
            if self.stop_by_time:
                return self._run_with_time_budget_ray(x_train, y_train)
            else:
                return self._run_with_max_cell_budget_ray(x_train, y_train)
        else:
            if self.stop_by_time:
                return self._run_with_time_budget(x_train, y_train)
            else:
                return self._run_with_max_cell_budget(x_train, y_train)

    def refit_and_predict(self, x_train, y_train, x_test):
        # refit and score
        self.best_model.refit(x_train, y_train)
        y_pred = self.best_model.predict(x_test)
        return y_pred

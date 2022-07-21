# -*- coding: utf-8 -*-
import numpy as np
from timeout_decorator import TimeoutError

from core.edeas.search_space.cp_space import EASpace
from core.pdeas.model_manager.model_manager_classification import Cascade as CascadeCLS
from core.pdeas.model_manager.model_manager_ray import Cascade as CascadeRay
from core.pdeas.model_manager.model_manger_regression import Cascade as CascadeREG
from core.utils.constants import CLASSIFIERS, REGRESSORS
from core.utils.helper import index2model
from core.utils.log_util import create_logger

logger = create_logger()


class DAGSpace(EASpace):
    def __init__(self, X_train, Y_train, X_val=None, Y_val=None, max_cell=8, random_state=None, kth=None,
                 task='classification', is_parallel=False, confidence_screen=False):
        # model count
        self.X_train = X_train
        self.Y_train = Y_train
        self.Y_val = Y_val
        self.X_val = X_val
        self.cell_dim = max_cell
        self.random_state = random_state
        self.kth = kth
        self.task = task
        self.is_parallel = is_parallel
        self.confidence_screen = confidence_screen

        if self.task == 'classification':
            self.estimators = CLASSIFIERS
        elif self.task == 'regression':
            self.estimators = REGRESSORS
        else:
            raise ValueError("Not supported task.")

        name = "Evolution based deep ensemble model search"
        M = 1
        maxormins = [-1]
        self.Dim = self.cell_dim * 2
        varTypes = [1 for _ in range(self.Dim)]

        lb = [0, 0]
        ub = [1, len(self.estimators) - 1]

        for i in range(1, self.cell_dim):
            lb.extend([-1, 0])
            ub.extend([i, len(self.estimators) - 1])

        lbin = [1 for _ in range(self.Dim)]
        ubin = [1 for _ in range(self.Dim)]

        super().__init__(name, M, maxormins, self.Dim, varTypes, lb, ub, lbin, ubin)

        # record best model and best score
        self.best_model = None
        self.best_score = 0
        self.model_count = 0

    def _aimFunc_cls(self, pop):
        models = pop.Phen
        m = len(models)
        new_models = []
        for i in range(m):
            new_model = index2model(models[i], task=self.task)
            new_models.append(new_model)

        scores = np.zeros(shape=(m, 1))
        for i in range(m):
            try:
                if self.kth is None:
                    logger.info(f"Current model: {new_models[i]}.")
                else:
                    logger.info(f"{self.kth + 1}-fold, current model: {new_models[i]}.")
                cas = CascadeCLS(child=new_models[i], cell_time_limit=120, confidence_screen=self.confidence_screen)
                cas.fit(self.X_train, self.Y_train, self.X_val, self.Y_val)
                cur_score = cas.best_score
            except (ValueError, TimeoutError):
                logger.info("Time out.")
                cur_score = 0
            except:
                logger.info("Running failed on this dataset.")
                cur_score = 0
            self.model_count += 1
            if self.kth is None:
                logger.info(f"Model {self.model_count}: {cur_score}.\n")
            else:
                logger.info(f"{self.kth + 1}-fold, model {self.model_count} score: {cur_score}.\n")
            if cur_score > self.best_score:
                self.best_score = cur_score
                self.best_model = cas
            scores[i][0] = cur_score
        pop.ObjV = scores

    def _aimFunc_reg(self, pop):
        models = pop.Phen
        m = len(models)
        new_models = []
        for i in range(m):
            new_model = index2model(models[i], task='regression')
            new_models.append(new_model)

        scores = np.zeros(shape=(m, 1))
        for i in range(m):
            try:
                if self.kth is None:
                    logger.info(f"Current model: {new_models[i]}.")
                else:
                    logger.info(f"{self.kth + 1}-fold, current model: {new_models[i]}.")
                cas = CascadeREG(child=new_models[i], cell_time_limit=120, confidence_screen=self.confidence_screen)
                cas.fit(self.X_train, self.Y_train, self.X_val, self.Y_val)
                cur_score = cas.best_score
            except (ValueError, TimeoutError):
                logger.info("Time out.")
                cur_score = 0
            except Exception as e:
                logger.info("Running failed on this dataset.")
                cur_score = 0
            self.model_count += 1
            if self.kth is None:
                logger.info(f"Model {self.model_count}: {cur_score}.\n")
            else:
                logger.info(f"{self.kth + 1}-fold, model {self.model_count} score: {cur_score}.\n")
            if cur_score > self.best_score:
                self.best_score = cur_score
                self.best_model = cas
            scores[i][0] = cur_score
        pop.ObjV = scores

    def _aimFunc_parallel(self, pop):
        models = pop.Phen
        m = len(models)
        new_models = []
        for i in range(m):
            new_model = index2model(models[i], self.task)
            new_models.append(new_model)

        scores = np.zeros(shape=(m, 1))
        for i in range(m):
            try:
                if self.kth is None:
                    logger.info(f"Current model: {new_models[i]}.")
                else:
                    logger.info(f"{self.kth + 1}-fold, current model: {new_models[i]}.")
                cas = CascadeRay(child=new_models[i], cell_time_limit=120, task=self.task)
                cas.fit(self.X_train, self.Y_train, self.X_val, self.Y_val)
                cur_score = cas.best_score
            except TimeoutError:
                logger.info("Time out.")
                cur_score = 0
            # except:
            #     logger.info("Running failed on this dataset.")
            #     cur_score = 0
            self.model_count += 1
            if self.kth is None:
                logger.info(f"Model {self.model_count}: {cur_score}.\n")
            else:
                logger.info(f"{self.kth + 1}-fold, model {self.model_count} score: {cur_score}.\n")
            if cur_score > self.best_score:
                self.best_score = cur_score
                self.best_model = cas
            scores[i][0] = cur_score
        pop.ObjV = scores

    def aimFunc(self, pop):
        if self.is_parallel:
            return self._aimFunc_parallel(pop)
        else:
            if self.task == 'classification':
                return self._aimFunc_cls(pop)
            elif self.task == 'regression':
                return self._aimFunc_reg(pop)
            else:
                raise ValueError("Not supported task.")

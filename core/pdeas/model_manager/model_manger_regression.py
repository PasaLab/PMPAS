# todo 更加优雅的方式来实现这些代码
import numpy as np
from sklearn.datasets import load_boston
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ARDRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold, train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import LinearSVR
from sklearn.svm import NuSVR
from sklearn.tree import DecisionTreeRegressor
from timeout_decorator import timeout
from xgboost import XGBRegressor, XGBRFRegressor

from core.utils.data_util import load_data_by_id
from core.utils.log_util import create_logger

logger = create_logger()


class MGS:
    def __init__(self, feature_shape, config=None, stride_ratio=1 / 16, sampling=None):
        """
        :param config: Estimator configuration of the multi-graned scanning layer.
        :param feature_shape: Int  or list or tuple or numpy.ndarray, if int means sequence data else means image data.
        :param stride_ratio: Ratio of the sliding window.
        """

        if isinstance(feature_shape, int):
            feature_shape = [feature_shape]
        self.feature_shape = np.array(feature_shape)
        self.config = config
        self.stride_ratio = stride_ratio
        self.sampling = sampling
        self._init_window()
        if len(self.feature_shape) == 1:
            logger.info(f"Stride ratio: {self.stride_ratio}, Sliding window size: {self.window}")
        else:
            logger.info(f"Stride ratio: {self.stride_ratio}, Sliding window size: [{self.window}x{self.window}]")

        self._init_config()
        logger.debug("Multi-grained scannning initializing finished...")

    def _init_window(self):
        self.window = int(self.feature_shape[0] * self.stride_ratio)
        self.stride = 1

    def _init_config(self):
        self.rf = RandomForestClassifier(n_estimators=20, max_features='sqrt', oob_score=True)
        self.erf = RandomForestClassifier(n_estimators=20, max_features=1, oob_score=True)

    def slicing(self, X, y=None):
        """
        :param X:
        :param y:
        :return:
        """
        if len(self.feature_shape) == 1:
            logger.info("Slicing sequence data...")
            return self._slicing_sequence(X, y)
        elif len(self.feature_shape) == 2:
            logger.info("Slicing image data...")
            return self._slicing_image(X, y)

    def _slicing_sequence(self, X, y=None):
        if self.window == 0:
            raise ValueError("Window size is 0, which is too small!")
        if self.window > self.feature_shape[0]:
            raise ValueError("Window size too large!")

        len_iter = np.floor_divide((self.feature_shape[0] - self.window), self.stride) + 1
        iter_array = np.arange(0, self.stride * len_iter, self.stride)

        ind_1X = np.arange(np.prod(self.feature_shape))
        inds_to_take = [ind_1X[i:i + self.window] for i in iter_array]
        # if self.sampling is not None:
        #     inds_to_take = np.random.choice(len(inds_to_take), max(1, int(self.sampling * len(inds_to_take))),
        #                                     replace=False)
        if self.sampling is not None:
            sampled_indices = np.random.choice(len(inds_to_take), max(1, int(len(inds_to_take) * self.sampling)),
                                               replace=False)
            inds_to_take = np.take(inds_to_take, sampled_indices, axis=0)
        sliced_sqce = np.take(X, inds_to_take, axis=1).reshape(-1, self.window)
        if y is not None:
            sliced_target = np.repeat(y, len(inds_to_take))
            return sliced_sqce, sliced_target
        else:
            return sliced_sqce

    def _slicing_image(self, X, y=None):
        if any(x < self.window for x in self.feature_shape):
            raise ValueError("The window to slicing image is too large!")
        len_iter_x = np.floor_divide((self.feature_shape[1] - self.window), self.stride) + 1
        len_iter_y = np.floor_divide((self.feature_shape[0] - self.window), self.stride) + 1
        iterx_array = np.arange(0, self.stride * len_iter_x, self.stride)
        itery_array = np.arange(0, self.stride * len_iter_y, self.stride)
        ref_row = np.arange(0, self.window)
        ref_ind = np.ravel([ref_row + self.feature_shape[1] * i for i in range(self.window)])
        inds_to_take = [ref_ind + ix + self.feature_shape[1] * iy
                        for ix, iy in np.itertools.product(iterx_array, itery_array)]
        sliced_imgs = np.take(X, inds_to_take, axis=1).reshape(-1, self.window ** 2)

        if y is not None:
            sliced_target = np.repeat(y, len_iter_x * len_iter_y)
            return sliced_imgs, sliced_target
        else:
            return sliced_imgs

    def fit_transform(self, X, y):
        """
        :param X:
        :param y:
        :return:
        """
        sliced_X, sliced_y = self.slicing(X, y)
        logger.info(f"Sliding window transform training X: {X.shape} into shape: {sliced_X.shape}")
        self.rf.fit(sliced_X, sliced_y)
        self.erf.fit(sliced_X, sliced_y)
        concat_prob = np.hstack([self.rf.oob_decision_function_, self.erf.oob_decision_function_])
        transformed_X = concat_prob.reshape(X.shape[0], -1)
        logger.info(f"After multi-grained scanning, transform training X into shape: {transformed_X.shape}")
        return transformed_X

    def predict_transform(self, X):
        sliced_X = self.slicing(X)
        logger.info(f"Sliding window transform test X: {X.shape} into shape: {sliced_X.shape}")
        rf_prob = self.rf.predict_proba(sliced_X)
        erf_prob = self.erf.predict_proba(sliced_X)
        concat_prob = np.hstack([rf_prob, erf_prob])
        transformed_X = concat_prob.reshape(X.shape[0], -1)
        logger.info(f"After multi-grained scanning, transform test X into shape: {transformed_X.shape}")
        return transformed_X


class Cascade:
    """
    Cascade layer of deep forest.
    Implemented by sklearn-style API.
    """

    # todo, modify the representation of cv
    def __init__(self, child, k_fold=3, max_layers=float("inf"), tolerance=0, early_stopping_rounds=3,
                 random_state=None, cell_time_limit=None, scoring='r2', confidence_screen=False):
        """

        :param child: The string representation of the model. e.g. ['svc',0,'svc',0,'svc',2]
        [-1] means the previous output (if exists), [2] means from the current layer output
        :param k_fold: Using k-fold validation strategy.
        :param max_layers: Maximum layers of the cascade layer.
        :param tolerance: Controlling the growth of the cascade layer.
        :param early_stopping_rounds: If the score of the layer does not change in some rounds, we stop it early.
        :param random_state: Random state passed by the caller, mainly used for experiment reproduction, currently
        :param cell_time_limit: Time limit (seconds) for each cell, None or -1 means no time constraint.
        :param scoring: Score function for evaluation.
        not been used.
        """
        self.child = child
        self.k_fold = k_fold
        # todo,回归问题的部分代码需要修改，尤其是这儿的cv部分
        self.cv = 3
        self.max_layers = max_layers
        # todo choose between tolerance and early_stopping_rounds
        self.tolerance = tolerance
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.cell_time_limit = cell_time_limit
        self.scoring = scoring
        self.confidence_screen = confidence_screen

        self.classes = 0
        self.best_layer_id = 0
        self.best_score = float("-inf")

        # store best models
        self.results = []
        self.refit_results = []

    def __repr__(self):
        return self.child

    def parse_child(self):
        """
        [operator,index,operator,index]
        Parsing child, separating it into indexes and operators
        :return:
        """
        indexes, operators_str = self.child[::2], self.child[1::2]
        return indexes, operators_str

    def init_operator(self, operator_str):
        """
        Initializing the sklearn estimator from string representation of the operator.
        :return:
        """
        if operator_str == 'ARDRegression':
            return ARDRegression()
        if operator_str == 'AdaBoostRegressor':
            return AdaBoostRegressor()
        if operator_str == 'BaggingRegressor':
            return BaggingRegressor()
        if operator_str == 'DecisionTreeRegressor':
            return DecisionTreeRegressor()
        if operator_str == 'ExtraTreesRegressor':
            return ExtraTreesRegressor()
        if operator_str == 'GradientBoostingRegressor':
            return GradientBoostingRegressor()
        if operator_str == 'KNeighborsRegressor':
            return KNeighborsRegressor()
        if operator_str == 'LinearSVR':
            return LinearSVR()
        if operator_str == 'MLPRegressor':
            return MLPRegressor()
        if operator_str == 'NuSVR':
            return NuSVR()
        if operator_str == 'RandomForestRegressor':
            return RandomForestRegressor()
        if operator_str == 'Ridge':
            return Ridge()
        if operator_str == 'SGDRegressor':
            return SGDRegressor()
        if operator_str == 'XGBRegressor':
            return XGBRegressor()
        if operator_str == 'XGBRFRegressor':
            return XGBRFRegressor()

    def fit(self, X, y, X_val=None, y_val=None):
        if not self.confidence_screen:
            # todo choose between train acc and val acc
            logger.info("Confidence screen mechanism off.")
            if X_val is None and y_val is None:
                logger.info(
                    "Fit process of cascade model, no validation data passed, training terminated by training score.")
                x_train, y_train = X, y
                self.classes = np.unique(y)
                preserved = self.dag_compute()
                indexes, operator_strs = self.parse_child()
                layer_id = 1
                prev_train_outputs = []
                while True:
                    operators = [self.init_operator(operator_str) for operator_str in operator_strs]
                    train_outputs = []
                    train_outputs_all = []
                    for (i, (operator, index)) in enumerate(zip(operators, indexes)):
                        if index == 0:
                            if layer_id == 1:
                                new_x_train = x_train
                            else:
                                new_x_train = np.column_stack([x_train, *prev_train_outputs])
                        else:
                            new_x_train = np.column_stack([x_train, train_outputs_all[index - 1]])
                        logger.debug(
                            f"[layer {layer_id}], fit {operator_strs[i]} on X: {new_x_train.shape}, y: {y_train.shape}.")
                        new_x_train = np.nan_to_num(new_x_train)
                        # todo remove this line?
                        train_proba = self.operator_fit(new_x_train, operator, y_train)
                        train_outputs_all.append(train_proba)
                        if preserved[i]:
                            train_outputs.append(train_proba)
                    prev_train_outputs = train_outputs
                    y_train_pred = np.mean(train_outputs, axis=0)
                    layer_score = r2_score(y_train, y_train_pred)
                    logger.info(f"[layer {layer_id}], training score: {layer_score}.")
                    self.results.append(operators)
                    if layer_id == 1:
                        layer_id += 1
                        prev_score = layer_score
                        self.best_layer_id = 1
                        self.best_score = layer_score
                        continue
                    if layer_id >= self.max_layers:
                        logger.info("Maximum layers reached.")
                        break
                    if layer_score <= prev_score + self.tolerance:
                        logger.info("Training score doesn't increase.")
                        break
                    prev_score = layer_score
                    if layer_score > self.best_score:
                        self.best_score = layer_score
                        self.best_layer_id = layer_id
                    logger.info(f"Best layer: {self.best_layer_id}, best score: {self.best_score}.")
                    layer_id += 1
            else:
                logger.info(
                    "Fit process of cascade model, validation data passed, training terminated by validation score.")
                x_train, x_val, y_train, y_val = X, X_val, y, y_val
                self.classes = np.unique(y)
                preserved = self.dag_compute()
                indexes, operator_strs = self.parse_child()
                layer_id = 1
                prev_train_outputs = []
                prev_val_outputs = []
                while True:
                    operators = [self.init_operator(operator_str) for operator_str in operator_strs]
                    train_outputs = []
                    train_outputs_all = []
                    for (i, (operator, index)) in enumerate(zip(operators, indexes)):
                        if index == 0:
                            if layer_id == 1:
                                new_x_train = x_train
                            else:
                                new_x_train = np.column_stack([x_train, *prev_train_outputs])
                        else:
                            new_x_train = np.column_stack([x_train, train_outputs_all[index - 1]])
                        logger.debug(
                            f"[layer {layer_id}], fit {operator_strs[i]} on X: {new_x_train.shape}, y: {y_train.shape}.")
                        new_x_train = np.nan_to_num(new_x_train)
                        # todo remove this line?
                        train_proba = self.operator_fit(new_x_train, operator, y_train)
                        train_outputs_all.append(train_proba)
                        if preserved[i]:
                            train_outputs.append(train_proba)
                    prev_train_outputs = train_outputs

                    val_outputs = []
                    val_outputs_all = []
                    for (i, (operator, index)) in enumerate(zip(operators, indexes)):
                        if index == 0:
                            if layer_id == 1:
                                new_x_val = x_val
                            else:
                                new_x_val = np.column_stack([x_val, *prev_val_outputs])
                        else:
                            new_x_val = np.column_stack([x_val, val_outputs_all[index - 1]])
                        logger.debug(
                            f"[layer {layer_id}], validate {operator_strs[i]} on X: {new_x_val.shape}.")
                        # todo remove this line?
                        new_x_val = np.nan_to_num(new_x_val)
                        val_proba = operator.predict(new_x_val)
                        val_outputs_all.append(val_proba)
                        if preserved[i]:
                            val_outputs.append(val_proba)
                    prev_val_outputs = val_outputs
                    y_train_pred = np.mean(train_outputs, axis=0)
                    y_val_pred = np.mean(val_outputs, axis=0)
                    train_score = r2_score(y_train, y_train_pred)
                    layer_score = r2_score(y_val, y_val_pred)
                    logger.info(f"[layer {layer_id}], training score: {train_score}.")
                    logger.info(f"[layer {layer_id}], validation score: {layer_score}.")
                    self.results.append(operators)
                    if layer_id == 1:
                        layer_id += 1
                        prev_score = layer_score
                        self.best_layer_id = 1
                        self.best_score = layer_score
                        continue
                    if layer_id >= self.max_layers:
                        logger.info("Maximum layers reached.")
                        break
                    if layer_score <= prev_score + self.tolerance:
                        logger.info("Validation score doesn't change.")
                        break
                    prev_score = layer_score
                    self.best_score = layer_score
                    self.best_layer_id = layer_id
                    if layer_score > self.best_score:
                        self.best_score = layer_score
                        self.best_layer_id = layer_id
                    logger.info(f"Best layer: {self.best_layer_id}, best score: {self.best_score}.")
                    layer_id += 1
        else:
            return self.fit_cs(X, y, X_val, y_val)

    def operator_refit(self, new_x_train, operator, y_train):
        return self._operator_fit(new_x_train, operator, y_train)

    def operator_fit(self, new_x_train, operator, y_train):
        return timeout(self.cell_time_limit, exception_message="Time limit for current cell has reached.")(
            self._operator_fit)(new_x_train, operator, y_train)

    def _operator_fit(self, new_x_train, operator, y_train):
        operator.fit(new_x_train, y_train)
        train_proba = cross_val_predict(operator, new_x_train, y_train, cv=self.cv,
                                        method='predict')
        return train_proba

    # todo bug when the layer grow too much, scores decrease immediately.
    def fit_cs(self, X, y, X_val=None, y_val=None, mgs=False):
        logger.info("Confidence screen mechanism on.")
        if X_val is None and y_val is None:
            logger.info(
                "Fit process of cascade model, no validation data passed, training terminated by training score.")
            x_train, y_train = X, y
            self.classes = np.unique(y)
            preserved = self.dag_compute()
            indexes, operator_strs = self.parse_child()
            layer_id = 1
            prev_train_outputs = []
            while True:
                operators = [self.init_operator(operator_str) for operator_str in operator_strs]
                train_outputs = []
                train_outputs_all = []
                for (i, (operator, index)) in enumerate(zip(operators, indexes)):
                    if index == 0:
                        if layer_id == 1:
                            new_x_train = x_train
                        else:
                            new_x_train = np.column_stack([x_train, *prev_train_outputs])
                    else:
                        new_x_train = np.column_stack([x_train, train_outputs_all[index - 1]])
                    logger.debug(
                        f"[layer {layer_id}], fit {operator_strs[i]} on X: {new_x_train.shape}, y: {y_train.shape}.")
                    new_x_train = np.nan_to_num(new_x_train)
                    train_proba = self.operator_fit(new_x_train, operator, y_train)
                    train_outputs_all.append(train_proba)
                    if preserved[i]:
                        train_outputs.append(train_proba)
                prev_train_outputs = train_outputs
                y_train_pred = np.mean(train_outputs, axis=0)
                layer_score = r2_score(y_train, y_train_pred)
                logger.info(f"[layer {layer_id}], training score: {layer_score}.")
                self.results.append(operators)
                if layer_id == 1:
                    layer_id += 1
                    prev_score = layer_score
                    self.best_layer_id = 1
                    self.best_score = layer_score
                    if mgs:
                        a = 1 / 20
                    else:
                        if layer_score > 0.9:
                            a = 1 / 10
                        else:
                            a = 1 / 3
                    continue
                new_preserved = self._confidence_screen(np.mean(train_outputs, axis=0), y_train, a)
                x_train = x_train[new_preserved]
                y_train = y_train[new_preserved]

                prev_train_outputs = [t[new_preserved, :] for t in train_outputs]
                if x_train.shape[0] < self.k_fold:
                    logger.info("Too few instances passed to next layer.")
                    break
                if layer_id >= self.max_layers:
                    logger.info("Maximum layers reached.")
                    break
                if layer_score <= prev_score + self.tolerance:
                    logger.info("Training score doesn't change.")
                    break
                prev_score = layer_score
                self.best_score = layer_score
                self.best_layer_id = layer_id
                if layer_score > self.best_score:
                    self.best_score = layer_score
                    self.best_layer_id = layer_id
                logger.info(f"Best layer: {self.best_layer_id}, best score: {self.best_score}.")
                layer_id += 1
        else:
            logger.info(
                "Fit process of cascade model, validation data passed, training terminated by validation score.")
            x_train, x_val, y_train, y_val = X, X_val, y, y_val
            self.classes = np.unique(y)
            preserved = self.dag_compute()
            indexes, operator_strs = self.parse_child()
            layer_id = 1
            prev_train_outputs = []
            prev_val_outputs = []
            while True:
                operators = [self.init_operator(operator_str) for operator_str in operator_strs]
                train_outputs = []
                train_outputs_all = []
                for (i, (operator, index)) in enumerate(zip(operators, indexes)):
                    if index == 0:
                        if layer_id == 1:
                            new_x_train = x_train
                        else:
                            new_x_train = np.column_stack([x_train, *prev_train_outputs])
                    else:
                        new_x_train = np.column_stack([x_train, train_outputs_all[index - 1]])
                    logger.debug(
                        f"[layer {layer_id}], fit {operator_strs[i]} on X: {new_x_train.shape}, y: {y_train.shape}.")
                    new_x_train = np.nan_to_num(new_x_train)
                    # todo remove this line?
                    train_proba = self.operator_fit(new_x_train, operator, y_train)
                    train_outputs_all.append(train_proba)
                    if preserved[i]:
                        train_outputs.append(train_proba)
                prev_train_outputs = train_outputs

                val_outputs = []
                val_outputs_all = []
                for (i, (operator, index)) in enumerate(zip(operators, indexes)):
                    if index == 0:
                        if layer_id == 1:
                            new_x_val = x_val
                        else:
                            new_x_val = np.column_stack([x_val, *prev_val_outputs])
                    else:
                        new_x_val = np.column_stack([x_val, val_outputs_all[index - 1]])
                    logger.debug(
                        f"[layer {layer_id}], validate {operator_strs[i]} on X: {new_x_val.shape}.")
                    new_x_val = np.nan_to_num(new_x_val)
                    val_proba = operator.predict(new_x_val)
                    val_outputs_all.append(val_proba)
                    if preserved[i]:
                        val_outputs.append(val_proba)
                prev_val_outputs = val_outputs
                y_train_pred = np.mean(train_outputs, axis=0)
                y_val_pred = np.mean(val_outputs, axis=0)
                train_score = r2_score(y_train, y_train_pred)
                layer_score = r2_score(y_val, y_val_pred)
                logger.info(f"[layer {layer_id}], training score: {train_score}.")
                logger.info(f"[layer {layer_id}], validation score: {layer_score}.")
                self.results.append(operators)
                if layer_id == 1:
                    layer_id += 1
                    prev_score = layer_score
                    self.best_layer_id = 1
                    self.best_score = layer_score
                    if mgs:
                        a = 1 / 20
                    else:
                        if layer_score > 0.9:
                            a = 1 / 10
                        else:
                            a = 1 / 3
                    continue
                # todo, regression task doesn't support confidence screen
                new_preserved = self._confidence_screen(np.array(train_outputs), y_train, a)
                x_train = x_train[new_preserved]
                y_train = y_train[new_preserved]

                prev_train_outputs = [t[new_preserved, :] for t in train_outputs]

                if x_train.shape[0] < self.k_fold:
                    logger.info("Too few instances passed to next layer.")
                    break
                if layer_id >= self.max_layers:
                    logger.info("Maximum layers reached.")
                    break
                if layer_score <= prev_score + self.tolerance:
                    logger.info("Validation score doesn't increase.")
                    break
                prev_score = layer_score
                self.best_score = layer_score
                self.best_layer_id = layer_id
                if layer_score > self.best_score:
                    self.best_score = layer_score
                    self.best_layer_id = layer_id
                logger.info(f"Best layer: {self.best_layer_id}, best score: {self.best_score}.")
                layer_id += 1

    def refit(self, X, y):
        """
        Refit can used by fit and fit with confidence screen.
        :param X:
        :param y:
        :return:
        """
        if self.best_layer_id == 0:
            raise Exception("You must fit before refit.")
        x_train, y_train = X, y
        self.classes = np.unique(y)
        preserved = self.dag_compute()
        indexes, operator_strs = self.parse_child()
        prev_train_outputs = []
        for layer_id, operators in enumerate(self.results[:self.best_layer_id], 1):
            train_outputs = []
            train_outputs_all = []
            for (i, (operator, index)) in enumerate(zip(operators, indexes)):
                if index == 0:
                    if layer_id == 1:
                        new_x_train = x_train
                    else:
                        new_x_train = np.column_stack([x_train, *prev_train_outputs])
                else:
                    new_x_train = np.column_stack([x_train, train_outputs_all[index - 1]])
                logger.debug(
                    f"[layer {layer_id}], refit {operator_strs[i]} on X: {new_x_train.shape}, y: {y_train.shape}.")
                new_x_train = np.nan_to_num(new_x_train)
                # todo remove this line?
                train_proba = self.operator_refit(new_x_train, operator, y_train)
                train_outputs_all.append(train_proba)
                if preserved[i]:
                    train_outputs.append(train_proba)
            prev_train_outputs = train_outputs
            logger.info(f"Refitting [layer {layer_id}] finished.")

    def _confidence_screen(self, X, y, a):
        m, n = X.shape
        preserved = np.array([True] * m)
        eps = 0
        for i in range(m):
            if y[i] != X[i].argmax():
                eps += 1
        eps /= m
        indices = X.max(axis=1).argsort()[::-1]
        L = [1] * (m + 1)
        res = []
        for k in range(1, m + 1):
            inc = 1 if X[indices[k - 1]].argmax() != y[[indices[k - 1]]] else 0
            L[k] = (k - 1) * L[k - 1] / k + inc / k
            if L[k] < a * eps:
                res.append(X[indices[k - 1]].max())
        eta = 1 if not res else min(res)
        new_X, new_y = [], []
        for i in range(m):
            if X[i].max() <= eta:
                new_X.append(X[i])
                new_y.append(y[i])
        for i in range(m):
            if X[i].max() > eta:
                preserved[i] = False
        return preserved

    def dag_compute(self):
        """
        compute which node to output, using heuristic method
        compute from back to front, if Y depend on X, then X doesn't output.
        :return:
        """
        indexes, _ = self.parse_child()
        n = len(indexes)
        preserved = [1 for _ in range(n)]
        for i in range(n - 1, -1, -1):
            assert indexes[i] in range(i + 1), "Invalid model passed, the i-th model index must be in [0...i-1]"
            if not preserved[i]:
                continue
            j = indexes[i]
            while j > 0:
                preserved[j - 1] = 0
                j = indexes[j - 1]
        return preserved

    def predict(self, X_test):
        """
        Predict the label of the given un-know samples.
        :param X_test:
        :return:
        """
        preserved = self.dag_compute()
        indexes, operator_strs = self.parse_child()
        prev_test_outputs = []
        for layer_id, operators in enumerate(self.results[:self.best_layer_id], 1):
            test_outputs = []
            test_outputs_all = []
            for (i, (operator, index)) in enumerate(zip(operators, indexes)):
                if index == 0:
                    if layer_id == 1:
                        new_X_test = X_test
                    else:
                        new_X_test = np.column_stack([X_test, *prev_test_outputs])
                else:
                    new_X_test = np.column_stack([X_test, test_outputs_all[index - 1]])
                logger.debug(
                    f"[layer {layer_id}], predict {operator_strs[i]} on X: {new_X_test.shape}.")
                new_X_test = np.nan_to_num(new_X_test)
                test_proba = operator.predict(new_X_test)
                test_outputs_all.append(test_proba)
                if preserved[i]:
                    test_outputs.append(test_proba)
            prev_test_outputs = test_outputs
        return np.mean(test_outputs, axis=0)

    def get_child(self):
        return self.child

# todo, add verbose param to controlling logging behaviour
class Model:
    """
    Overall process of the deep ensemble architecture.
    """

    def __init__(self, cas_config, mgs_config=None, mgs=False, feature_shape=None,
                 sampling=None, model_time_limit=None, cell_time_limit=None):
        """
        :param cas_config:
        :param mgs_config:
        :param mgs: Whether to use multi-grain scanning, default False.
        :param confidence_screen:
        :param feature_shape:
        """
        # todo 增加置信度筛选机制,同时修改这个函数
        self.cas_config = cas_config
        self.mgs_config = mgs_config
        self.mgs = mgs
        self.feature_shape = feature_shape
        self.sampling = sampling
        self.cell_time_limit = cell_time_limit
        self.model_time_limit = model_time_limit

        if self.mgs:
            if self.feature_shape is None:
                raise Exception("When using multi-grained scanning, must provide the shape of the feature.")
            self.mgs_model = MGS(feature_shape, mgs_config, sampling=sampling)
        self.cas_model = Cascade(cas_config, cell_time_limit=cell_time_limit)

    # todo, more elegant way
    def fit(self, X, y, X_val=None, y_val=None, confidence_screen=False):
        """
        Fitting with time controlling.
        :param X:
        :param y:
        :param X_val:
        :param y_val:
        :param confidence_screen:
        :return:
        """
        timeout(self.model_time_limit, exception_message="Time limit for the current model has reached.") \
            (self._fit)(X, y, X_val, y_val, confidence_screen)

    def _fit(self, X, y, X_val=None, y_val=None, confidence_screen=False):
        if self.mgs:
            if X_val is None and y_val is None:
                X = self.mgs_model.fit_transform(X, y)
            else:
                transfomed_X = self.mgs_model.fit_transform(np.concatenate([X, X_val], axis=0),
                                                            np.concatenate([y, y_val], axis=0))
                X, X_val = transfomed_X[:len(X)], transfomed_X[len(X):]
        if confidence_screen:
            self.cas_model.fit_cs(X, y, X_val, y_val)
        else:
            self.cas_model.fit(X, y, X_val, y_val)

    def refit(self, X, y):
        self.cas_model.refit(X, y)

    def predict(self, X):
        if self.mgs:
            X = self.mgs_model.predict_transform(X)
        return self.cas_model.predict(X)

    # todo modify the string representation of the model
    def __repr__(self):
        return f'Deep forest model'


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")
    np.random.seed(0)
    import random

    random.seed(0)
    X, y = load_boston(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size=0.2)
    # child = [0, 'KNeighborsClassifier', 1, 'SVC', 2, 'LogisticRegression', 0, 'RandomForestClassifier', 1,
    #          'ExtraTreesClassifier', 2, 'AdaBoostClassifier', 3, 'DecisionTreeClassifier', 6, 'SGDClassifier', 4,
    #          'QuadraticDiscriminantAnalysis', 3, 'LinearDiscriminantAnalysis', 8, 'MultinomialNB', 2,
    #          'DecisionTreeClassifier',
    #          2, 'DecisionTreeClassifier', 0, 'LogisticRegression', 1, 'SVC']
    # model = [0, 'RandomForestClassifier', 0, 'RandomForestClassifier', 0, 'RandomForestClassifier', 0,
    #          'RandomForestClassifier', 0, 'ExtraTreesClassifier', 0, 'ExtraTreesClassifier', 0,
    #          'ExtraTreesClassifier',
    #          0, 'ExtraTreesClassifier']
    model = [0,
             'ARDRegression',
             1,
             'AdaBoostRegressor',
             2,
             'BaggingRegressor',
             3,
             'DecisionTreeRegressor',
             4,
             'ExtraTreesRegressor',
             5,
             'GradientBoostingRegressor',
             6,
             'KNeighborsRegressor',
             7,
             'LinearSVR',
             8,
             'MLPRegressor',
             9,
             'NuSVR',
             10,
             'RandomForestRegressor',
             11,
             'Ridge',
             12,
             'SGDRegressor',
             13,
             'XGBRegressor',
             14,
             'XGBRFRegressor']
    # model = RandomForestRegressor()
    # scores = cross_val_score(model, X, y, scoring='r2')
    # print(np.mean(scores))

    model = [0, 'XGBRFRegressor']
    c = Cascade(model)
    logger.info("Begin training...")
    # c.fit(x_train, y_train)
    c.fit(X_train, Y_train, X_val, Y_val)
    logger.info("Begin refitting...")
    c.refit(x_train, y_train)
    logger.info("Begin testing...")

    logger.info(f"op {r2_score(y_test, c.predict(x_test))}")
    #
    # cf = XGBRFRegressor()
    # cf.fit(x_train, y_train)
    # y_pred = cf.predict(x_test)
    # logger.info(f"new op {r2_score(y_test, y_pred)}")

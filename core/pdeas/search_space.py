from collections import OrderedDict

import numpy as np
import ray

from core.pdeas.model_manager.model_manager_ray import Cascade
from core.search_space import SearchSpace
from core.utils.log_util import create_logger

logger = create_logger()


@ray.remote
class Evaluator(object):
    def __init__(self, child, model_time_limit=None, cell_time_limit=120, task='classification'):
        self.child = child
        self.reward = 0
        self.model = None
        self.task = task

        self.model_time_limit = model_time_limit
        self.cell_time_limit = cell_time_limit

        pass

    def evaluate(self, X_train, Y_train, X_val, Y_val):
        logger.info(self.child)
        self.model = Cascade(child=self.child, task=self.task,
                             cell_time_limit=self.cell_time_limit)
        cur_score = 0
        try:
            self.model.fit(X_train, Y_train, X_val, Y_val)
        except TimeoutError:
            logger.info("Time limit for the current model has reached.")
        except:
            logger.info(f"{self.model} running failed.")
        cur_score = max(cur_score, self.model.best_score)
        logger.info(f"Current model score: {cur_score}.")
        self.reward = cur_score

    def return_two_value(self):
        return self.reward, self.model

    def get_score(self):
        return self.reward

    def get_model(self):
        return self.model


class CPSpace(SearchSpace):
    def __init__(self, B, estimators):
        super().__init__(estimators)
        self.states = OrderedDict()
        self.state_count_ = 0

        self.history = None
        self.intermediate_children = None

        self.B = B

        input_values = list(range(0, self.B))  # -1 = Hc-1, 0-(B-1) = Hci

        self.inputs_embedding_max = len(input_values)
        self.operator_embedding_max = len(np.unique(self.estimators))

        self._add_state('inputs', values=input_values)
        self._add_state('ops', values=self.estimators)
        self.generate_initial_models()

    def __repr__(self):
        return 'CP'

    def _add_state(self, name, values):
        index_map = {}
        for i, val in enumerate(values):
            index_map[i] = val

        value_map = {}
        for i, val in enumerate(values):
            value_map[val] = i

        metadata = {
            'id': self.state_count_,
            'name': name,
            'values': values,
            'size': len(values),
            'index_map_': index_map,
            'value_map_': value_map,
        }
        self.states[self.state_count_] = metadata
        self.state_count_ += 1

        return self.state_count_ - 1

    def embedding_encode(self, id, value):
        state = self[id]
        value_map = state['value_map_']
        value_idx = value_map[value]

        encoding = np.zeros((1, 1), dtype=np.float32)
        encoding[0, 0] = value_idx
        return encoding

    def get_state_value(self, id, index):
        state = self[id]
        index_map = state['index_map_']
        value = index_map[index]
        return value

    def parse_state_space_list(self, state_list):
        state_values = []
        for id, state_value in enumerate(state_list):
            state_val_idx = state_value[0, 0]
            value = self.get_state_value(id % 2, state_val_idx)
            state_values.append(value)

        return state_values

    def entity_encode_child(self, child):
        encoded_child = []
        for i, val in enumerate(child):
            encoded_child.append(self.embedding_encode(i % 2, val))

        return encoded_child

    def generate_initial_models(self):
        ops = list(range(len(self.estimators)))

        inputs = [0]

        search_space = [inputs, ops]
        self.history = list(self._construct_permutations(search_space))

    def generate_new_models(self, new_b):
        new_b_dash = self.construct_structure()

        new_ip_values = list(range(0, new_b_dash))
        ops = list(range(len(self.estimators)))

        # if input_lookback_depth == 0, then we need to adjust to have at least
        # one input (generally 0)
        if len(new_ip_values) == 0:
            new_ip_values = [0]

        search_space = [new_ip_values, ops]
        new_search_space = list(self._construct_permutations(search_space))

        for i, child in enumerate(self.history):
            for permutation in new_search_space:
                temp_child = list(child)
                temp_child.extend(permutation)
                yield temp_child

    def construct_structure(self):
        return 1

    def _construct_permutations(self, search_space):
        for input1 in search_space[0]:
            for operation1 in search_space[1]:
                yield [input1, self.estimators[operation1]]

    def update_children(self, children):
        self.history = children

    def __getitem__(self, id):
        return self.states[id % self.size]

    @property
    def size(self):
        return self.state_count_

    def print_total_models(self, K):
        """
        Compute the total number of models to generate and train
        :param K: beam search arg
        :return:
        """
        num_inputs = 1
        level1 = (num_inputs ** 2) * (len(self.estimators) ** 2)
        remainder = (self.B - 1) * K
        total = level1 + remainder
        return total


class DAGSpace(SearchSpace):
    def __init__(self, B, estimators):
        super().__init__(estimators)

        self.states = OrderedDict()
        self.state_count_ = 0

        self.history = None
        self.intermediate_children = None

        self.B = B

        input_values = list(range(0, self.B))

        self.inputs_embedding_max = len(input_values)
        self.operator_embedding_max = len(np.unique(self.estimators))

        self._add_state('inputs', values=input_values)
        self._add_state('ops', values=self.estimators)
        self.generate_initial_models()

    # todo xiugai biaoshi fangfa
    def __repr__(self):
        return "DAG"

    def _add_state(self, name, values):
        index_map = {}
        for i, val in enumerate(values):
            index_map[i] = val

        value_map = {}
        for i, val in enumerate(values):
            value_map[val] = i

        metadata = {
            'id': self.state_count_,
            'name': name,
            'values': values,
            'size': len(values),
            'index_map_': index_map,
            'value_map_': value_map,
        }
        self.states[self.state_count_] = metadata
        self.state_count_ += 1

        return self.state_count_ - 1

    def embedding_encode(self, id, value):
        state = self[id]
        value_map = state['value_map_']
        value_idx = value_map[value]

        encoding = np.zeros((1, 1), dtype=np.float32)
        encoding[0, 0] = value_idx
        return encoding

    def get_state_value(self, id, index):
        state = self[id]
        index_map = state['index_map_']
        value = index_map[index]
        return value

    def parse_state_space_list(self, state_list):
        state_values = []
        for id, state_value in enumerate(state_list):
            state_val_idx = state_value[0, 0]
            value = self.get_state_value(id % 2, state_val_idx)
            state_values.append(value)

        return state_values

    def entity_encode_child(self, child):
        encoded_child = []
        for i, val in enumerate(child):
            encoded_child.append(self.embedding_encode(i % 2, val))

        return encoded_child

    def generate_initial_models(self):
        ops = list(range(len(self.estimators)))

        inputs = [0]

        search_space = [inputs, ops]
        self.history = list(self._construct_permutations(search_space))

    def generate_new_models(self, new_b):
        new_b_dash = self.construct_structure(new_b)

        new_ip_values = list(range(0, new_b_dash))
        ops = list(range(len(self.estimators)))

        # if input_lookback_depth == 0, then we need to adjust to have at least
        # one input (generally 0)
        if len(new_ip_values) == 0:
            new_ip_values = [0]

        search_space = [new_ip_values, ops]
        new_search_space = list(self._construct_permutations(search_space))

        for i, child in enumerate(self.history):
            for permutation in new_search_space:
                temp_child = list(child)
                temp_child.extend(permutation)
                yield temp_child

    def construct_structure(self, new_b):
        return new_b

    def _construct_permutations(self, search_space):
        for input1 in search_space[0]:
            for operation1 in search_space[1]:
                yield [input1, self.estimators[operation1]]

    def update_children(self, children):
        self.history = children

    def __getitem__(self, id):
        return self.states[id % self.size]

    @property
    def size(self):
        return self.state_count_

    # todo修改这个函数的功能
    def print_total_models(self, K):
        num_inputs = 1
        level1 = (num_inputs ** 2) * (len(self.estimators) ** 2)
        remainder = (self.B - 1) * K
        total = level1 + remainder

        print("Total number of models : ", total)
        print()
        return total

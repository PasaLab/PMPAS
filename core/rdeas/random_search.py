import numpy as np

from core.utils.constants import ESTIMATORS
from core.utils.constants import REGRESSORS
from core.utils.helper import model2valid


# todo ,抽象出next接口来实现下一个模型的获取,抽象出一个基类从而让子类来实现相应的方法,用一种统一的方式来修改max cell num 这个参数，最好是写在
#  常数模块里面
class RandomSearch:
    def __init__(self, search_space='DAG', max_cell_num=8, task='classification'):
        """
        Random search from the user-specified search space.
        :param search_space: DAG or Plain
        :param max_cell_num: Maximum cell nums
        """
        self.search_space = search_space
        self.max_cell_num = max_cell_num
        self.task = task

    def next(self):
        model = []
        if self.task == 'classification' or self.task == 'cls':
            N_ESTIMATORS = len(ESTIMATORS)
            if self.search_space == 'DAG':
                for i in range(self.max_cell_num):
                    if i == 0:
                        index = 0
                        cell_index = np.random.random_integers(0, N_ESTIMATORS - 1)
                    else:
                        index = np.random.random_integers(-1, i)
                        cell_index = np.random.random_integers(0, N_ESTIMATORS - 1)
                    model.append(index)
                    model.append(ESTIMATORS[cell_index])
            else:
                # todo, 未测试结果
                for i in range(self.max_cell_num):
                    if i == 0:
                        cell_index = 0
                    else:
                        cell_index = np.random.random_integers(-1, N_ESTIMATORS - 1)
                    model.append(0)
                    model.append(ESTIMATORS[cell_index])
        elif self.task == 'regression' or self.task == 'reg':
            N_ESTIMATORS = len(REGRESSORS)
            if self.search_space == 'DAG':
                for i in range(self.max_cell_num):
                    if i == 0:
                        index = 0
                        cell_index = np.random.random_integers(0, N_ESTIMATORS - 1)
                    else:
                        index = np.random.random_integers(-1, i)
                        cell_index = np.random.random_integers(0, N_ESTIMATORS - 1)
                    model.append(index)
                    model.append(REGRESSORS[cell_index])
            else:
                # todo, 未测试结果
                for i in range(self.max_cell_num):
                    if i == 0:
                        cell_index = 0
                    else:
                        cell_index = np.random.random_integers(-1, N_ESTIMATORS - 1)
                    model.append(0)
                    model.append(REGRESSORS[cell_index])
        else:
            raise ValueError("Not valid model.")
            pass
        valid_model = model2valid(model, self.task)
        return valid_model


if __name__ == '__main__':
    while True:
        rs = RandomSearch(task='regression')
        cur_model = rs.next()
        print(cur_model)

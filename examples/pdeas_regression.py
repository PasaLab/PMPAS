from api.auto_del import SearchSpace, SearchAlgorithm, AutoDEL
from core.utils.data_util import load_data

X_train, X_test, y_train, y_test = load_data(505, task='regression')

search_space = SearchSpace(space_type='DAG')

search_algorithm = SearchAlgorithm(method='PMPAS', K=4, task='regression')

auto_del = AutoDEL(search_space=search_space,
                   search_algorithm=search_algorithm,
                   task='regression',
                   budget_type='running_time',
                   total_budget=120,
                   random_state=26)

auto_del.experiment_summary()

auto_del.search_with_algorithm(X_train, y_train)

y_pred = auto_del.refit_and_predict(X_train, y_train, X_test)

score = auto_del.score(y_pred, y_test)

auto_del.result_summary()

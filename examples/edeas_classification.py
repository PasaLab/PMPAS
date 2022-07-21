from api.auto_del import SearchSpace, SearchAlgorithm, AutoDEL
from core.utils.data_util import load_data

# load dataset
# when passing classification dataset, just passing dataset id
# when passing regression dataset, also need to passing parameter task='regression'
X_train, X_test, y_train, y_test = load_data(11)

# set search space
# space can be 'CP' or 'DAG'
search_space = SearchSpace(space_type='CP')

# set search algorithm
# search algorithm can be 'EPEAAS' or 'PMPAS'
search_algorithm = SearchAlgorithm(method='EPEAAS')

# set PASA-AutoDEL class
# passing experiment argument
auto_del = AutoDEL(search_space=search_space,
                   search_algorithm=search_algorithm,
                   task='classification',
                   budget_type='running_time',
                   total_budget=240,
                   random_state=26)

# experiment summary
# currently just showing important information
# can be extend to show more information
auto_del.experiment_summary()

# search process, must provide dataset
auto_del.search_with_algorithm(X_train, y_train)

# refit and predict process, must provide dataset
y_pred = auto_del.refit_and_predict(X_train, y_train, X_test)

# score(predict score)
score = auto_del.score(y_pred, y_test)

# result summary
auto_del.result_summary()

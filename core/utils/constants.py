B = 8  # 每层最大得cell数量
K = 8  # 每次迭代训练的子模型
KFOLD = 3  # 交叉验证的折数
PER_MODEL_RUNTIME_LIMIT = 120

MAX_CELL = 8  # 进化算法和随机搜索的最大cell数量

MAX_EPOCHS = 8  # 训练的最大epoch
BATCHSIZE = 256  # batch size
CHILD_MODEL_LR = 0.0001  # 子模型的学习率
REGULARIZATION = 0  # 正则化强度
CONTROLLER_CELLS = 64  # RNN控制器unit数量
RNN_TRAINING_EPOCHS = 8  # 训练控制器的epoch数量
RESTORE_CONTROLLER = False  # 恢复控制器以继续训练

# 默认的搜索空间
# todo 修改这个和模型会不一致,需要同时修改
ESTIMATORS = ['AdaBoostClassifier',
              'BernoulliNB',
              'DecisionTreeClassifier',
              'ExtraTreesClassifier',
              'GaussianNB',
              'GaussianProcessClassifier',
              'GradientBoostingClassifier',
              'KNeighborsClassifier',
              'LinearDiscriminantAnalysis',
              'LogisticRegression',
              'MLPClassifier',
              'MultinomialNB',
              'QuadraticDiscriminantAnalysis',
              'RandomForestClassifier',
              'SGDClassifier',
              'SVC']

CLASSIFIERS = ['AdaBoostClassifier',
               'BernoulliNB',
               'DecisionTreeClassifier',
               'ExtraTreesClassifier',
               'GaussianNB',
               'GaussianProcessClassifier',
               'GradientBoostingClassifier',
               'KNeighborsClassifier',
               'LinearDiscriminantAnalysis',
               'LogisticRegression',
               'MLPClassifier',
               'MultinomialNB',
               'QuadraticDiscriminantAnalysis',
               'RandomForestClassifier',
               'SGDClassifier',
               'SVC']

REGRESSORS = [  # 'ARDRegression', 这个太慢了
    'AdaBoostRegressor',
    'BaggingRegressor',
    'DecisionTreeRegressor',
    'ExtraTreesRegressor',
    'GradientBoostingRegressor',
    'KNeighborsRegressor',
    'LinearSVR',
    'MLPRegressor',
    'NuSVR',
    'RandomForestRegressor',
    'Ridge',
    # 'SGDRegressor', 这个分数总是小于0
    'XGBRegressor',
    'XGBRFRegressor']





# mini dataset
DATASET_IDS = [3, 6, 11, 12, 14, 15, 16, 18, 22, 23, 28, 29, 31, 32, 37, 44, 46, 50, 54, 151, 182, 188, 38, 307, 300,
               458, 469, 1049, 1050, 1053, 1063, 1067, 1068, 1590, 1510, 1489, 1494, 1497, 1501, 1480, 1485, 1486, 1487,
               1468, 1475, 1462, 1464, 4534, 6332, 1461, 4538, 1478, 23381, 40499, 40668, 40966, 40982, 40994, 40983,
               40975, 40984, 40979, 41027, 23517, 40670, 40701]



# lack 181
best_ids = [6, 12, 14, 16, 18, 22, 23, 28, 32, 50, 54, 182, 300, 458, 1462, 1501, 4534, 23381, 40499, 40670, 40983,
            1590]




# mini dataset
# 27
REGRESSION_IDS = [201, 216, 287, 416, 422, 505, 507, 531, 541, 546, 550, 574, 3050, 3277, 41021, 41540, 41702, 41980,
                  42225, 42563, 42570, 42688, 42705, 42724, 42726, 42730, 42731]





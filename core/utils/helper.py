from core.utils.constants import ESTIMATORS, REGRESSORS


def validate(model):
    """
    Validate whether a model is valid.
    :param model:
    :return:
    """
    n = len(model)
    if n % 2 != 0:
        raise ValueError("Not valid model.")
    n //= 2
    indexes = model[::2]
    for i in range(n - 1, -1, -1):
        if indexes[i] not in range(i + 1):
            raise ValueError("Not valid model.")


def split_model(child):
    indexes, operators_str = child[::2], child[1::2]
    return indexes, operators_str


def index2model(pop, task='classification'):
    """
    transform evolutionary intermediate indexes to real model
    :param pop:
    :return:
    """
    if task == 'classification':
        indexes, operators_str = split_model(pop)
        new_model = [0, ESTIMATORS[operators_str[0]]]
        cnt = 1
        for i in range(1, len(indexes)):
            if indexes[i] == -1 or indexes[indexes[i] - 1] == -1:
                indexes[i] = -1
                continue
            if indexes[i] == 0:
                new_model.append(0)
            else:
                new_model.append(cnt)
                cnt += 1
            new_model.append(ESTIMATORS[operators_str[i]])
    elif task == 'regression' or task == 'REGRESSION':
        indexes, operators_str = split_model(pop)
        new_model = [0, REGRESSORS[operators_str[0]]]
        cnt = 1
        for i in range(1, len(indexes)):
            if indexes[i] == -1 or indexes[indexes[i] - 1] == -1:
                indexes[i] = -1
                continue
            if indexes[i] == 0:
                new_model.append(0)
            else:
                new_model.append(cnt)
                cnt += 1
            new_model.append(REGRESSORS[operators_str[i]])
    else:
        raise ValueError("Not supported task!")
    return new_model


def model2valid(model, task='classification'):
    """
    transform evolutionary intermediate indexes to real model
    :param pop:
    :return:
    """
    if task == 'classification':
        indexes, operators_str = split_model(model)
        new_model = [0, operators_str[0]]
        cnt = 1
        for i in range(1, len(indexes)):
            if indexes[i] == -1 or indexes[indexes[i] - 1] == -1:
                indexes[i] = -1
                continue
            if indexes[i] == 0:
                new_model.append(0)
            else:
                new_model.append(cnt)
                cnt += 1
            new_model.append(operators_str[i])
    elif task == 'regression' or task == 'REGRESSION':
        indexes, operators_str = split_model(model)
        new_model = [0, operators_str[0]]
        cnt = 1
        for i in range(1, len(indexes)):
            if indexes[i] == -1 or indexes[indexes[i] - 1] == -1:
                indexes[i] = -1
                continue
            if indexes[i] == 0:
                new_model.append(0)
            else:
                new_model.append(cnt)
                cnt += 1
            new_model.append(operators_str[i])
    else:
        raise ValueError("Not supported task!")
    return new_model


# todo  for regression
def plain_encoding_to_model(encoding, task='classification'):
    n = len(encoding)
    model = []
    if task == 'classification':
        for i in range(n):
            model.append(0)
            model.append(ESTIMATORS[encoding[i]])
    else:
        for i in range(n):
            model.append(0)
            model.append(REGRESSORS[encoding[i]])
    return model

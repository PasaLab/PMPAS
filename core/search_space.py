class SearchSpace(object):
    def __init__(self, estimators=None, history=None):
        if estimators is not None:
            self.estimators = estimators
        self.history = history

    def set_estimators(self, new_estimators):
        self.estimators = new_estimators

    def get_estimators(self):
        return self.estimators

    def generate_initial_models(self):
        pass

    def construct_structure(self):
        pass

    def generate_new_models(self):
        pass

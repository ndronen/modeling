class NullPreprocessor(object):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        pass

    def transform(self, X, y=None):
        if y is None:
            return X
        else:
            return X, y

    def fit_transform(self, X, y=None):
        return self.transform(X, y)

from copy import deepcopy


class InputSpecificModel:
    def __init__(self, classifier, X_train, X_test, y_train, y_test):
        self.classifier = classifier
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def fit(self, X=None, y=None):
        if X is None and y is None:
            self.classifier.fit(self.X_train, self.y_train)
        elif (X is not None) and (y is not None):
            self.classifier.fit(X, y)
        else:
            raise ValueError("X and y must be both or neither specified")

        # returns self so that methods can be chained called like model.fit().score() or model.fit().predict()
        return self

    def predict(self, X=None):
        if X is None:
            return self.classifier.predict(X)
        else:
            return self.classifier.predict(self.X_test)

    def predict_proba(self, X=None):
        if X is None:
            return self.classifier.predict_proba(self.X_test)
        else:
            return self.classifier.predict_proba(X)

    def score(self, X=None, y=None):
        if X is None and y is None:
            return self.classifier.score(self.X_test, self.y_test)
        elif (X is not None) and (y is not None):
            return self.classifier.score(X, y)
        else:
            raise ValueError("X and y must be both or neither specified")

    def get_classifier(self):
        return deepcopy(self.classifier)

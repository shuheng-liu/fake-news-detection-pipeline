from sklearn.metrics import f1_score
from model.input_specific_model import InputSpecificModel as Voter

import numpy as np


class EnsembleVoter:
    def __init__(self, classifiers: list, Xs_train: list, Xs_test: list, y_train: np.ndarray, y_test: np.ndarray,
                 weights=None):
        """
        returns a model that ensemble classifiers with different input embeddings and the same output embeddings
        :param classifiers: list of classifier to be voting
        :param Xs_train: list[np.ndarray or scipy.sparse.csr_matrix] for training, corresponding to `classifiers`
        :param Xs_test: list[np.ndarray or scipt.sparse.csr_matrix] for testing, corresponding to `classifiers`
        :param y_train: np.ndarray, labels for training
        :param y_test: np.ndarray, labels for testing
        :param weights: iterable or `None`, weight for each voter
        """
        self.fitted = False
        self.voters = [Voter(*tup, y_train, y_test) for tup in zip(classifiers, Xs_train, Xs_test)]
        self.n_voters = len(self.voters)
        self.weights = self._set_weights(weights)
        self.proba = None
        self.y_test = y_test

    def _set_weights(self, weights):
        if weights is None:
            self.fit()
            return self._set_weights([voter.score() for voter in self.voters])
        else:
            total_weights = sum(weights)
            return [weight / total_weights for weight in weights]

    def fit(self, verbose=True, refit=False):
        if self.fitted and not refit:
            print("Fittng aborted because all voters are fitted and not using refit=True")
            return

        for voter in self.voters:
            voter.fit()
            if verbose:
                print("Test score of {}: {}".format(voter.classifier.__class__.__name__, voter.score()))

        self.fitted = True

    def score(self):
        return f1_score(self.y_test, self.predict())

    def predict(self):
        return np.argmax(self.predict_proba(), axis=1)

    def predict_proba(self):
        if self.proba is None:
            self.proba = np.sum(voter.predict_proba() * weight for voter, weight in zip(self.voters, self.weights))
        return self.proba


if __name__ == '__main__':
    from hypertuned_models import mlp, lg, svc, qda
    from embedding_loader import EmbeddingLoader
    from sklearn.model_selection import train_test_split

    loader = EmbeddingLoader("../pretrained/")
    d2v_500 = loader.get_d2v(corpus="concat", win_size=23, epochs=500)
    d2v_100 = loader.get_d2v(corpus="concat", win_size=13, epochs=100)
    onehot = loader.get_onehot(corpus="concat", scorer="tfidf")
    labels = loader.get_label()

    d2v_500_train, d2v_500_test, d2v_100_train, d2v_100_test, onehot_train, onehot_test, labels_train, labels_test = \
        train_test_split(d2v_500, d2v_100, onehot, labels, test_size=0.25, stratify=labels, random_state=11)

    classifiers = [mlp, svc, qda, lg]
    Xs_train = [d2v_500_train, d2v_100_train, d2v_100_train, onehot_train]
    Xs_test = [d2v_500_test, d2v_100_test, d2v_100_test, onehot_test]

    # classifiers = [mlp, svc, lg]
    # Xs_train = [d2v_500_train, d2v_100_train, onehot_train]
    # Xs_test = [d2v_500_test, d2v_100_test, onehot_test]

    # the variable tentative_weights is used for voting
    tentative_weights = [1, 1, 1, 1]
    ens_voter = EnsembleVoter(classifiers, Xs_train, Xs_test, labels_train, labels_test, tentative_weights)
    ens_voter.fit()
    print("Test score of EnsembleVoter: ", ens_voter.score())

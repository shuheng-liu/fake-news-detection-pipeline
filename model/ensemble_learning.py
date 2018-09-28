from mlxtend.classifier import EnsembleVoteClassifier
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.metrics import classification_report, f1_score
from sklearn import model_selection
from model.input_specific_model import InputSpecificModel as Voter

import numpy as np


class EnsembleLearning:
    def __init__(self, classifiers, labels):
        self.classifiers = classifiers
        self.labels = labels

    def ensemble_vote_classifier(self, x_train, y_train, x_test, y_test, weights_list=None, voting="soft"):
        if weights_list == None:
            weights_list = [1] * len(self.classifiers)

        # initialize the ensemble learning classifier
        eclf = EnsembleVoteClassifier(clfs=self.classifiers,
                                      weights=weights_list, voting=voting)

        # show the result of every classifier
        for clf, label in zip(self.classifiers, self.labels):
            scores = model_selection.cross_val_score(clf, x_train, y_train,
                                                     cv=5,
                                                     scoring='f1')
            print("F1-score: %0.3f (+/- %0.3f) [%s]"
                  % (scores.mean(), scores.std(), label))
        # show the result of ensemble voting
        scores = model_selection.cross_val_score(eclf, x_train, y_train, cv=5, scoring='f1')
        print("F1-score: %0.3f (+/- %0.3f) [%s]" % (scores.mean(), scores.std(), 'EnsembleVoting'))

        # build the ensemble model and make the prediction
        eclf.fit(x_train, y_train)
        y_pred = eclf.predict(x_test)

        # Report the metrics
        target_names = ['Real', 'Fake']
        print("EnsembleVoteClassifier:")
        print(classification_report(y_true=y_test, y_pred=y_pred, target_names=target_names, digits=3))

    def ensemble_weighted_voting(self, x_train, y_train, x_test, y_test, weights_list=None):
        y_pred = []
        for model in self.classifiers:
            model.fit(x_train, y_train)
            y_pred.append(model.predict(x_test))

        if weights_list == None:
            weights_list = [1] * len(self.classifiers)

        y_ensemble_pred = []
        for idx in range(len(x_test)):
            vote_sum = 0.0
            weight_sum = 0.0
            for itr in range(len(self.classifiers)):
                weight_sum += weights_list[itr]
                vote_sum += weights_list[itr] * y_pred[itr][idx]
            vote_sum = vote_sum / weight_sum
            # obtain the ensemble result
            if vote_sum >= 0.5:
                y_ensemble_pred.append(1)
            else:
                y_ensemble_pred.append(0)

        # Report the metrics
        target_names = ['Real', 'Fake']
        print("EnsembleWeightedVoting:")
        print(classification_report(y_true=y_test, y_pred=y_ensemble_pred, target_names=target_names, digits=3))


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
    from hypertuned import mlp, lg, svc, qda
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

    ens_voter = EnsembleVoter(classifiers, Xs_train, Xs_test, labels_train, labels_test)
    ens_voter.fit()
    print("Test score of EnsembleVoter: ", ens_voter.score())

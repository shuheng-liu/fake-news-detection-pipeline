from mlxtend.classifier import EnsembleVoteClassifier
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.metrics import classification_report
from sklearn import model_selection

import numpy as np


class EnsembleLearning:
    def __init__(self, classifiers, labels):
        self.classifiers = classifiers
        self.labels = labels

    def ensemble_vote_classifier(self, x_train, y_train, x_test, y_test, weights_list=None, voting=None):
        if weights_list == None:
            weights_list = [1] * len(self.classifiers)
        if voting == None:
            voting = 'soft'

        # initialize the ensemble learning classifer
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
        print("EnsembleVoteClassifer:")
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

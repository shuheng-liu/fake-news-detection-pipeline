from mlxtend.classifier import EnsembleVoteClassifier
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.metrics import classification_report
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import numpy as np

# MLP classifier
mlp = MLPClassifier(activation='relu', alpha=0.01, batch_size='auto', beta_1=0.8,
                    beta_2=0.9, early_stopping=False, epsilon=1e-08,
                    hidden_layer_sizes=(600, 300), learning_rate='constant',
                    learning_rate_init=0.0001, max_iter=200, momentum=0.9,
                    nesterovs_momentum=True, power_t=0.5, random_state=0, shuffle=True,
                    solver='adam', tol=0.0001, validation_fraction=0.1, verbose=False,
                    warm_start=False)

# KNN classifier
knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='cosine',
                           metric_params=None, n_jobs=1, n_neighbors=10, p=2,
                           weights='distance')

# QDA classifier
qda = QuadraticDiscriminantAnalysis(priors=np.array([0.5, 0.5]),
                                    reg_param=0.6531083254653984, store_covariance=False,
                                    store_covariances=None, tol=0.0001)

# GDB classifier
gdb = GradientBoostingClassifier(criterion='friedman_mse', init=None,
                                 learning_rate=0.1, loss='exponential', max_depth=10,
                                 max_features='log2', max_leaf_nodes=None,
                                 min_impurity_decrease=0.0, min_impurity_split=None,
                                 min_samples_leaf=0.0012436966435001434,
                                 min_samples_split=100, min_weight_fraction_leaf=0.0,
                                 n_estimators=200, presort='auto', random_state=0,
                                 subsample=0.8, verbose=0, warm_start=False)

# SVC classifier
svc = SVC(C=0.8, cache_size=200, class_weight=None, coef0=0.0,
          decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
          max_iter=-1, probability=False, random_state=0, shrinking=True,
          tol=0.001, verbose=False)

# GNB classifier
gnb = GaussianNB(priors=None)

# RF classifier
rf = RandomForestClassifier(bootstrap=False, class_weight=None,
                            criterion='entropy', max_depth=10, max_features=7,
                            max_leaf_nodes=None, min_impurity_decrease=0.0,
                            min_impurity_split=None, min_samples_leaf=9,
                            min_samples_split=6, min_weight_fraction_leaf=0.0,
                            n_estimators=50, n_jobs=1, oob_score=False, random_state=None,
                            verbose=0, warm_start=False)


# All the parameters of the classifiers above are optimal in our experiments
# The list below is used to store every classifier instance
classifiers_list = [mlp, knn, qda, gdb, svc, gnb, rf]
classifiers_labels = ['MultiplePercetron', 'KNeighbours', 'QuadraticDiscriminantAnalysis', 'GradientBoosting', 'SVC', 'GaussianNB', 'RandomForest']


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





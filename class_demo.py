#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'Administrator'
__mtime__ = '2018/9/16'
# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
              ┏┓      ┏┓
            ┏┛┻━━━┛┻┓
            ┃      ☃      ┃
            ┃  ┳┛  ┗┳  ┃
            ┃      ┻      ┃
            ┗━┓      ┏━┛
                ┃      ┗━━━┓
                ┃  神兽保佑    ┣┓
                ┃　永无BUG！   ┏┛
                ┗┓┓┏━┳┓┏┛
                  ┃┫┫  ┃┫┫
                  ┗┻┛  ┗┻┛
"""
from sklearn.model_selection import train_test_split
from functools import partial
from hyperopt import fmin, hp, tpe, Trials, space_eval
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np
from time import time
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

class LearningProcess:
    def __init__(self, X, y, **kwargs):
        """
        init a learning process
        :param X: features for training and testing
        :param y: labels for training and testing
        :param kwargs: other parameters for train-test-split
        """
        # split data into train and test sets
        seed = 0

        validation_size = .25
        test_size = .25
        train_validation_size = 1 - test_size
        validation_size_adjusted = validation_size / train_validation_size

        # we must split one more time
        self.X_train_validation, self.X_test, self.y_train_validation, self.y_test = train_test_split(X, y, \
                                                                                  test_size=test_size, \
                                                                                  random_state=seed)

        # perform the second split which splits the validation/test data into two distinct datasets
        self.X_train, self.X_validation, self.y_train, self.y_validation = train_test_split(self.X_train_validation, self.y_train_validation, \
                                                                        test_size=validation_size_adjusted, \
                                                                        random_state=seed)




    def try_model(self, model, cv=10, duration=60, **hyperparameters):
        best_result = {}
        start = time()

        random_search = RandomizedSearchCV(model, param_distributions=hyperparameters, cv=cv)
        while time() - start < duration:
            random_search.fit(self.X_train, self.y_train)

            # TODO update the best results
            pass

        return best_result

    def try_model_with_hpt(self, model, cv=10, times=300, **hyperparameters):
        def objective(params, X, y):
            self.clf = model
            self.clf.set_params(**params)

            score = cross_val_score(self.clf, X, y, scoring='f1', cv=cv).mean()
            print("F1 {:.3f} params {}".format(score, params))
            return score

        hyperopt_objective = lambda params: (-1.0) * objective(params, self.X_train_validation, self.y_train_validation)
        # Trail
        trials = Trials()

        # Set algoritm parameters
        algo = partial(tpe.suggest,
                       n_startup_jobs=20, gamma=0.25, n_EI_candidates=24)

        # Fit Tree Parzen Estimator
        best_vals = fmin(hyperopt_objective, space=hyperparameters,
                         algo=algo, max_evals=times, trials=trials,
                         rstate=np.random.RandomState(seed=0))

        # Print best parameters
        best_params = space_eval(hyperparameters, best_vals)
        print("BEST PARAMETERS: " + str(best_params))

        # Print best CV score
        scores = [-1 * trial['result']['loss'] for trial in trials.trials]
        print("BEST CV SCORE: " + str(np.max(scores)))

        # Print execution time
        tdiff = trials.trials[-1]['book_time'] - trials.trials[0]['book_time']
        print("Searching TIME: " + str(tdiff.total_seconds() / 60))

        self.clf.set_params(**best_params)
        self.clf.fit(self.X_train_validation, self.y_train_validation)
        test_score = self.clf.score(self.X_test,self.y_test)
        train_score = self.clf.score(self.X_train_validation,self.y_train_validation)
        print("Training accuracy is:{}".format(train_score))
        print("Testing accuracy is:{}".format(test_score))
        self.y_pred = self.clf.predict(self.X_test)
        print(classification_report(self.y_test, self.y_pred))
        train_f1 = np.max(scores)
        test_f1 = f1_score(self.y_test, self.y_pred)
        print('f1 score of positive samples:{}'.format(test_f1))


        return self.clf, str(best_params), train_score, test_score, train_f1, test_f1

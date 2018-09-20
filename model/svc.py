from scipy.stats import randint as sp_randint
from sklearn import svm

model = svm.SVC()
param_dist = {"C": [0.1, 0.2, 0.4, 0.8, 1.0, None],
              "kernel": ["linear", "poly", "rbf", "sigmoid"],
              "shrinking": [True, False],
              "decision_function_shape": ["ovo", "ovr"],
              "random_state": sp_randint(0,7)}
from sklearn.naive_bayes import GaussianNB
import numpy as np

model = GaussianNB()
param_dist = {"priors": [None, np.array([0.5, 0.5]), np.array([0.1, 0.9]),np.array([0.3, 0.7]), np.array([0.7, 0.3])]}
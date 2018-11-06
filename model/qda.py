import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from scipy.stats.distributions import uniform

model = QuadraticDiscriminantAnalysis(priors=np.array([0.5, 0.5]))

param_dist = {
    "reg_param": uniform(0, 1)
}

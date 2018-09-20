from sklearn.linear_model import LogisticRegression
from scipy.stats import randint, uniform

seed = 0
model = LogisticRegression()
param_dist = {
    # "penalty": ['l1', 'l2'],
    "penalty": ['l2'],

    # "C": [0.1, 0.5, 1.0, 2, 10],
    "C": uniform(0.001, 0.01),
    "random_state": [seed],
    "max_iter": randint(500, 1000),
}

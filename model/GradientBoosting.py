from sklearn.ensemble import GradientBoostingClassifier
from scipy.stats import randint
from scipy.stats.distributions import uniform

seed = 0
model = GradientBoostingClassifier(random_state=seed)
param_dist = {
    "loss": ['deviance', 'exponential'],
    "learning_rate": [0.1, 0.03, 0.3],
    "n_estimators": [50, 100, 200],
    "max_depth": randint(1, 13),
    "min_samples_split": [2, 10, 30, 50, 100],
    "min_samples_leaf": uniform(0.0002, 0.002),
    "subsample": [1.0, 0.9, 0.8],
    "max_features": ["auto", "log2", None, 20, 40, 60],
    "min_impurity_decrease": [0.1 * x for x in range(11)],
}

from scipy.stats import randint as sp_randint
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
param_dist = {"n_estimators": [10, 20, 30, 40, 50],
              "max_depth": [3, 5, 7, 9, 10, None],
              "max_features": sp_randint(1, 20),
              "min_samples_split": sp_randint(2, 12),
              "min_samples_leaf": sp_randint(1, 12),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}


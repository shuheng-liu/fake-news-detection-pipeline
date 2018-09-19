from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import randint

model = KNeighborsClassifier()
param_dist = {
    "n_neighbors": randint(5, 50),
    "weights": ['uniform', 'distance'],
    "metric": ['minkowski', 'cosine'],
}

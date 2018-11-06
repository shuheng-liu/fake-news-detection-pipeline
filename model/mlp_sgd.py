from sklearn.neural_network import MLPClassifier
from scipy.stats import uniform
# if mlp_config cannot be loaded, make sure to append "PROJECT_ROOT/model" in your `PYTHONPATH` variable
from mlp_config import sizes
from mlp_config import random_seed as seed

model = MLPClassifier()
param_dist = {
    "hidden_layer_sizes": sizes,
    "activation": ["logistic", "tanh", "relu"],
    "solver": ["sgd"],
    "alpha": [0, 1e-4, 1e-3, 1e-2],
    "learning_rate_init": [1e-4, 1e-3, 1e-2],
    "learning_rate": ["constant", "invscaling", "adaptive"],
    "power_t": uniform(0.4, 1),
    "random_state": [seed],
    "max_iter": [100, 200, 500],
    "momentum": [0.5, 0.8, 0.9, 0.99],
    "nesterovs_momentum": [True, False],
}

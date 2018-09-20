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
    "alpha": [0, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3],
    "learning_rate": ["constant", "invscaling", "adaptive"],
    "learning_rate_init": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2],
    "power_t": uniform(0.4, 1),
    "random_state": [seed],
    "max_iter": [100, 200, 500],
    "momentum": [0.5, 0.7, 0.8, 0.9, 0.99, 0.999],
    "nesterovs_momentum": [True, False],
}

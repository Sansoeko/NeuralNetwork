import numpy as np


def activation_function_logistic(x):
    x = np.clip(x, -1e2, 1e2)
    y = 1.0 / (1.0 + np.exp(-x))
    return y


def activation_function_binary_step(x):
    return 0 if x < 0 else 1


def activation_function_tanh(x):
    y = 2.0 / (1.0 + math.exp(-2 * x) - 1)
    return y

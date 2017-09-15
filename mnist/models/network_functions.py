import math


def activation_function_logistic(x):
    y = 1.0/(1.0+math.exp(-x))
    return y


def activation_function_tanh(x):
    y = 2.0/(1.0+math.exp(-2*x) - 1)
    return y

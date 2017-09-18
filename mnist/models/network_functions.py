import math


def activation_function_logistic(x):
    try:
        y = 1.0/(1.0+math.exp(-x))
    except OverflowError:
        if x > 0:
            y = 1.0
        else:
            y = 0.0
    return y


def activation_function_binary_step(x):
    return 0 if x < 0 else 1


def activation_function_tanh(x):
    y = 2.0/(1.0+math.exp(-2*x) - 1)
    return y

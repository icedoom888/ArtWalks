import numpy as np
import scipy

def logit_sampler(x):
    return scipy.special.logit(x)/15 + 0.5

def squeezed_logit_sampler(x):
    coefficients = [1.86030624e-12, 1.07142857e+01, -2.67857143e+01, 2.77261905e+01, -1.48035714e+01, 4.14880952e+00, 5.03549833e-15]
    cubic_function = np.poly1d(coefficients)

    return scipy.special.logit(cubic_function(x))/15 + 0.5

def sinh_sampler(x):
    return np.sinh(x*4 - 2)/7.27 + 0.5

def squeezed_sinh_sampler(x):
    coefficients = [1.86030624e-12, 1.07142857e+01, -2.67857143e+01, 2.77261905e+01, -1.48035714e+01, 4.14880952e+00, 5.03549833e-15]
    cubic_function = np.poly1d(coefficients)

    return np.sinh(cubic_function(x)*4 - 2)/7.5 + 0.5

def linear_sampler(x):
    return x

def get_weight_sampler(sampler_name):
    sampler_func = globals().get(sampler_name + '_sampler')
    if sampler_func is not None:
        return sampler_func
    else:
        raise ValueError(f"Unknown sampler name: {sampler_name}")

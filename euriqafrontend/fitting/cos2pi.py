import numpy as np
from oitg.fitting import FitBase

def parameter_initialiser(x, y, p):

    p['y0'] = np.mean(y)
    p['x0'] = 0
    p['a'] = (np.max(y) - np.min(y))/2
    p['n_periods'] = 1


def fitting_function(x, p):

    oneperiod = np.max(x) - np.min(x)
    y = p['a']*np.cos(2*np.pi*(x/oneperiod*p['n_periods']) - p['x0'])
    y += p['y0']
    return y


# Cosine with 'dump' initialiser
cos2pi = FitBase.FitBase(['x0', 'y0', 'a', 'n_periods'], fitting_function,
                      parameter_initialiser=parameter_initialiser)

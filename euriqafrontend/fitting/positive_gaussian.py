import numpy as np
from oitg.fitting import FitBase
import scipy as sp

def parameter_initialiser(x, y, p):

    p["y0"] = 0  # np.mean(y)

    # Estimate the sign of the gaussian
    dy_min = p["y0"] - np.min(y)
    dy_max = np.max(y) - p["y0"]

    # The peak should be always positive
    # p["a"] = dy_max
    # p["x0"] = x[np.argmax(y)]

    # Estimate the x0 using scipy find peaks
    pks, prop = sp.signal.find_peaks(y, prominence=0.1)
    if len(pks) < 1:
        p["a"] = dy_max
        p["x0"] = x[np.argmax(y)]
    else:
        if len(pks) > 1:
            dis_left = pks
            dis_right = len(y) - pks  # [len(y)-i for i in pks]
            dis_min = [dis_left[i] if dis_left[i] < dis_right[i] else dis_right[i] for i in range(0, len(pks))]
            ind = pks[np.argmax(dis_min)]
        else:
            ind = pks[0]

        p['a'] = y[ind]
        p['x0'] = x[ind]

    # Estimate the sigma
    # In most cases the this initial parameter is a good guess
    # since most data-sets are sampled so that this is the case
    p["sigma"] = (1 / 5) * (np.max(x) - np.min(x))


def fitting_function(x, p):

    y = np.sqrt(p["a"] * p["a"]) * np.exp(-0.5 * ((x - p["x0"]) / p["sigma"]) ** 2)
    y += p["y0"]

    return y


def derived_parameter_function(p, p_error):

    # Calculate the FWHM from the sigma
    p["fwhm"] = 2.35482 * p["sigma"]
    p_error["fwhm"] = 2.35482 * p_error["sigma"]

    return p, p_error


positive_gaussian = FitBase.FitBase(
    ["x0", "y0", "a", "sigma"],
    fitting_function,
    parameter_initialiser=parameter_initialiser,
    derived_parameter_function=derived_parameter_function,
)

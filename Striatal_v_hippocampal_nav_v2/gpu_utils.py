from numba import guvectorize, float64
import numpy as np


@guvectorize([(float64[:], float64[:], float64, float64, float64, float64, float64[:])], '(n),(n),(),(),(),()->(n)')
def bvc_activation_function(ds, thetas, pref_distance, pref_orientation, sigma_rad, sigma_ang, res):
    # calculate activation
    i = 0
    for d, theta in zip(ds, thetas):
        distance_term = np.exp(-(d - pref_distance) ** 2 / (2 * sigma_rad ** 2)) / np.sqrt(
            2 * np.pi * sigma_rad ** 2)
        angle_term = np.exp(-(theta - pref_orientation) ** 2 / (2 * sigma_ang ** 2)) / np.sqrt(
            2 * np.pi * sigma_ang ** 2)
        res[i] = distance_term * angle_term
        i += 1

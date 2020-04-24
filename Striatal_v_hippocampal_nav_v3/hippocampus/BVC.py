import numpy as np
import matplotlib.pyplot as plt
import os
from definitions import ROOT_FOLDER

import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()})

from hippocampus.fastBVC import Boundary, BVC, Environment


bvc_maps_folder = os.path.join(ROOT_FOLDER, 'data', 'bvc_maps')


class FastPlaceCell(object):
    """Lookup version
    """
    def __init__(self):
        self.n_inputs = constrained_poisson_sample(4, 4, 16)
        self.pref_distances = np.random.choice([81.0, 169.0, 265.0, 369.0, 482.5, 606.5, 800.], self.n_inputs)
        self.pref_orientations = np.random.choice(np.linspace(0, 354, 60), self.n_inputs)

    def compute_ratemap(self, env):
        input_activity = []
        for bvc in range(self.n_inputs):
            dist = self.pref_distances[bvc]
            orient = self.pref_orientations[bvc]
            fname = 'bvcmap_{}_distance{}_orientation{}.npy'.format(env.boundaries_present, int(dist), int(orient))
            bvc_map = np.load(os.path.join(bvc_maps_folder, fname))
            input_activity.append(bvc_map)
        pc_activity = self.compute_activity(input_activity)
        return pc_activity

    def compute_activity(self, input_activity):
        pc_activity = heaviside_step(np.power(np.prod(input_activity, axis=0), 1 / self.n_inputs))
        return pc_activity


class PlaceCell(object):
    """

    Each place cell was modelled as the geometric mean of n BVCs where n was drawn from a Poisson distribution
    constrained between 2 and 16 (lambda = 4 cells).
    """
    def __init__(self):
        self.n_inputs = constrained_poisson_sample(4, 2, 16)

        self.input_cells = [BVC() for _ in range(self.n_inputs)]

    def compute_activation(self, pos, env):
        """

        :return:
        """
        bvc_activity = [bvc.compute_activation_pixel(pos, env) for bvc in self.input_cells]
        pc_activity = heaviside_step(np.power(np.prod(bvc_activity), 1 / len(bvc_activity)))
        return pc_activity

    def compute_ratemap(self, xs, ys, env):
        rate_map = np.empty((len(xs), len(ys)))
        for ix, x in enumerate(xs):
            for iy, y in enumerate(ys):
                pos = np.array([x, y])
                activation = self.compute_activation(pos, env)
                rate_map[ix, iy] = activation
        return rate_map


def heaviside_step(x):
    positive = np.greater(x, 0)
    x[~positive] = 0
    return x


def constrained_poisson_sample(lam, min_value, max_value):
    """Return a poisson sample within a constrained interval.

    :param float max_value: Maximum possible value.
    :param float min_value: Minimum possible value.
    :param int lam: Lambda parameter (average number of events).
    :return: A sample from a Poisson distribution within a constrained interval.
    """
    sample = max_value + 1
    while sample > max_value or sample < min_value:
        sample = np.random.poisson(lam)
    return sample


if __name__ == '__main__':
    """
    Precompute and save the BVC maps to save computation time. We can just look them up afterwards. 
    """
    from tqdm import tqdm


    # Precompute and save BVC rate maps
    if not os.path.exists(bvc_maps_folder):
        os.makedirs(bvc_maps_folder)

    # Make square environment of 1m x 1m
    b_left = np.array(((0, 0), (0, 1000)), dtype=np.float64)
    b_top = np.array(((0, 1000), (1000, 1000)), dtype=np.float64)
    b_right = np.array(((1000, 0), (1000, 1000)), dtype=np.float64)
    b_bottom = np.array(((0, 0), (1000, 0)), dtype=np.float64)

    env = Environment([b_left, b_top, b_right, b_bottom])

    distances = [81.0, 169.0, 265.0, 369.0, 482.5, 606.5]
    directions_degrees = np.linspace(0, 360, 61)[:-1]
    directions_radians = np.radians(directions_degrees)

    r = np.arange(5, 1010, 10)

    for dist in tqdm(distances, desc='Distances'):
        for orient in tqdm(directions_radians, leave=False, desc='orientations'):
            bvc = BVC(dist, orient)
            rate_map = bvc.compute_ratemap(r, r, env)
            rate_map = rate_map / rate_map.max()
            fname = 'bvcmap_{}_distance{}_orientation{}.npy'.format('Square1x1', int(dist), int(np.round(np.degrees(orient))))
            np.save(os.path.join(bvc_maps_folder, fname), rate_map)

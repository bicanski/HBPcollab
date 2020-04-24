import numpy as np

import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()})

from hippocampus.fastBVC import Boundary
from hippocampus.geometry_utils import intersect_lines, subtended_angle_c, in_smallest_interval


class BVC(object):
    """Boundary Vector Cell. Parameters taken from Burgess et al.
    """

    def __init__(self, pref_distance=None, pref_orientation=None):
        self.beta = 1830
        self.sigma_0 = 122
        self.sigma_ang = 0.2
        if pref_distance is None:
            self.pref_distance = np.random.choice([81.0, 169.0, 265.0, 369.0, 482.5, 606.5])
        else:
            self.pref_distance = pref_distance
        if pref_orientation is None:
            self.pref_orientation = np.radians(np.random.choice(np.linspace(0,354, 60)))
        else:
            self.pref_orientation = pref_orientation
        self.sigma_rad = (self.pref_distance / self.beta + 1) * self.sigma_0

    def distance_to_nearest_boundary_py(self, pos, direction, env):
        d = [np.inf]
        subtended_angle = []
        for b in env.boundaries:
            v1 = b.p1 - pos
            v2 = b.p2 - pos
            a1 = np.arctan2(v1[1], v1[0]) % (2 * np.pi)
            a2 = np.arctan2(v2[1], v2[0]) % (2 * np.pi)

            if in_smallest_interval(direction, a1, a2):
                d.append(b.distance_in_orientation(pos, direction))
                subtended_angle.append(b.subtended_angle(np.array(pos, dtype=np.float64)))
        return min(d), subtended_angle[d.index(min(d)) - 1]

    def distance_to_nearest_boundary(self, pos, orientation, env):
        idx = self.which_boundary(pos, orientation, env)
        b = env.boundaries[idx]
        d = b.distance_in_orientation(pos, orientation)
        a = b.subtended_angle(pos)
        return d, a

    def which_boundary(self,  pos, orientation, env):
        for i in range(env.n_boundaries):
            b = env.boundaries[i]
            v1 = b.p1 - pos
            v2 = b.p2 - pos
            a1 = np.arctan2(v1[1], v1[0]) % (2 * np.pi)
            a2 = np.arctan2(v2[1], v2[0]) % (2 * np.pi)
            if in_smallest_interval(orientation, a1, a2):
                return i
        return False

    def compute_activation_pixel(self, pos, env):
        angles = np.linspace(0, 2 * np.pi, 400)[:-1]
        n_angles = len(angles)
        ds = np.empty(len(angles), dtype=np.float64)

        for i in range(n_angles):
            theta = angles[i]

            # get distance and subtended angle
            d, subtended_angle = self.distance_to_nearest_boundary(pos, theta, env)
            f = self.calculate_activation(d, subtended_angle, theta)
            ds[i] = f
        return ds.sum()

    def compute_ratemap_grid(self, xs, ys, env):
        nx = len(xs)
        ny = len(ys)
        rate_map = np.zeros(nx, dtype=np.float64)

        for i, j in zip(range(nx), range(ny)):
            pos = np.array([xs[i], ys[j]], dtype=np.float64)
            activation = self.compute_activation_pixel(pos, env)
            rate_map[i] = activation
        return rate_map

    def calculate_activation(self, d, subtended_angle, theta):
        # calculate activation
        distance_term = np.exp(-(d - self.pref_distance) ** 2 / (2 * self.sigma_rad ** 2)) / np.sqrt(
            2 * np.pi * self.sigma_rad ** 2)
        angle_term = np.exp(-(theta - self.pref_orientation) ** 2 / (2 * self.sigma_ang ** 2)) / np.sqrt(
            2 * np.pi * self.sigma_ang ** 2)
        f = distance_term * angle_term * subtended_angle
        return f


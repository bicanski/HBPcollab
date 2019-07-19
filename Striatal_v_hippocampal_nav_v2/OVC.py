import numpy as np
from shapely.geometry import LineString, Point
from shapely import speedups

from multiprocessing import Pool
from itertools import repeat
from utils import in_circle
from gpu_utils import bvc_activation_function
from Experiments.Blocking.environment import Environment
import os


speedups.enable()
ovc_maps_folder = './Data/OVC_ratemaps/'

"""
class Environment(object):
    def __init__(self, cues='both'):
        self.xs = np.arange(5, 1010, 10)
        self.ys = np.arange(5, 1010, 10)

        object_1 = np.array([750, 650])
        object_2 = np.array([750, 700])
        if cues == 'both':
            self.objects = [object_1, object_2]
        elif cues == 'A':
            self.objects = [object_1]
        elif cues == 'B':
            self.objects = [object_2]
        self.cue_radius = 40
"""

class ObjectVectorCell(object):

    def __init__(self, pref_distance=None, pref_orientation=None):
        self.beta = 1830  # mm
        self.sigma_0 = 70  # mm
        self.sigma_ang = 0.3  # radians
        if pref_distance is None:
            self.pref_distance = np.random.choice([169.0, 265.0, 369.0, 482.5])
        else:
            self.pref_distance = pref_distance
        if pref_orientation is None:
            self.pref_orientation = np.radians(np.random.choice(np.linspace(0, 354, 60)))
        else:
            self.pref_orientation = pref_orientation
        self.sigma_rad = (self.pref_distance / self.beta + 1) * self.sigma_0

    def compute_activation_pixel(self, pos, env):
        for obj in env.objects:
            if in_circle(pos, obj, env.cue_radius):
                return 0.
        angles = np.linspace(0, 2 * np.pi, 800)[:-1]
        activations = np.empty(len(angles))
        for i, theta in enumerate(angles):
            # get distance and subtended angle
            d = self.distance_to_nearest_object(theta, pos, env)
            f = self.calculate_activation(d, theta)
            activations[i] = f
        return activations.sum()

    def _compute_activation_pixel(self, pos, env):
        # GPU version.
        for obj in env.objects:
            if in_circle(pos, obj, env.cue_radius):
                return 0.
        angles = np.linspace(0, 2 * np.pi, 800)[:-1]
        #activations = np.empty(len(angles))
        distances = np.empty(len(angles))
        for i, theta in enumerate(angles):
            # get distance and subtended angle
            distances[i] = self.distance_to_nearest_object(theta, pos, env)
        activations = bvc_activation_function(distances, angles, self.pref_distance, self.pref_orientation,
                                              self.sigma_rad, self.sigma_ang)
        return activations.sum()

    def distance_to_nearest_object(self, theta, pos, env):
        distances = [self.distance_to_circle(pos, theta, obj, env.cue_radius) for obj in env.objects]
        return min(distances)

    def calculate_activation(self, d, theta):
        # calculate activation
        distance_term = np.exp(-(d - self.pref_distance) ** 2 / (2 * self.sigma_rad ** 2)) / np.sqrt(
            2 * np.pi * self.sigma_rad ** 2)
        angle_term = np.exp(-(theta - self.pref_orientation) ** 2 / (2 * self.sigma_ang ** 2)) / np.sqrt(
            2 * np.pi * self.sigma_ang ** 2)
        f = distance_term * angle_term
        return f

    def _compute_ratemap(self, xs, ys, env):
        rate_map = np.zeros((len(xs), len(ys)))

        for i, x in enumerate(tqdm(xs)):
            for j, y in enumerate(ys):
                pos = np.array([xs[i], ys[j]], dtype=np.float64)
                activation = self.compute_activation_pixel(pos, env)
                rate_map[i, j] = activation
        return rate_map

    def compute_ratemap(self, xs, ys, env):
        # Uses multi processing
        rate_map = np.zeros((len(xs), len(ys)))

        p = Pool(len(ys))
        for i, x in enumerate(tqdm(xs)):
            positions = [(x, y) for y in ys]
            rate_map[i, :] = p.starmap(self.compute_activation_pixel, zip(positions, repeat(env)))
        p.close()
        p.join()
        return rate_map

    @staticmethod
    def distance_to_circle(pos, theta, circle_pos, radius):
        """Give the distance to a circle in a given direction

        :param pos:
        :param theta:
        :param circle_pos:
        :param radius:
        :return:
        """
        pix = Point(pos)
        l = LineString([pos, (pos[0] + 1000 * np.cos(theta), pos[1] + 1000 * np.sin(theta))])
        p = Point(circle_pos[0], circle_pos[1])
        c = p.buffer(radius).boundary
        i = c.intersection(l)
        if isinstance(i, Point):
            distance = pix.distance(i)
        elif len(i.geoms) > 0:
            distance = min([pix.distance(ip) for ip in i.geoms])
        else:
            distance = 10 ** 5
        return distance


if __name__ == '__main__':
    from tqdm import tqdm

    # Precompute and save BVC rate maps
    if not os.path.exists(ovc_maps_folder):
        os.makedirs(ovc_maps_folder)

    cue_list = ['both', 'A', 'B', 'C']

    for cue in tqdm(cue_list, desc='Cues'):

        # Make square environment of 1m x 1m
        env = Environment(cue)

        distances = [169.0, 265.0, 369.0, 482.5]
        directions_degrees = np.linspace(0, 360, 61)[:-1]
        directions_radians = np.radians(directions_degrees)

        r = np.arange(5, 1010, 10)

        for dist in tqdm(distances, desc='Distances', leave=False):
            for orient in tqdm(directions_radians, leave=False, desc='orientations'):
                fname = 'ovcmap_{}_distance{}_orientation{}.npy'.format(cue, int(dist),
                                                                        int(np.round(np.degrees(orient))))
                if os.path.isfile(os.path.join(ovc_maps_folder, fname)):
                    tqdm.write('skipping {}'.format(fname))
                    continue
                ovc = ObjectVectorCell(dist, orient)
                rate_map = ovc.compute_ratemap(r, r, env)
                rate_map = rate_map / rate_map.max()
                np.save(os.path.join(ovc_maps_folder, fname), rate_map)

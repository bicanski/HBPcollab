import numpy as np
cimport numpy as np

from libc.math cimport sqrt, acos, tan, atan2, M_PI, exp

from geometry_utils import intersect_lines, subtended_angle_c, in_smallest_interval


class Environment(object):
    def __init__(self, boundary_segments):
        self.boundaries = []
        for b in boundary_segments:
            boundary = Boundary(b[0], b[1])
            self.boundaries.append(boundary)
        self.n_boundaries = len(self.boundaries)


cdef class Boundary(object):
    #cdef public np.float64_t[:] p1
    #cdef public np.float64_t[:] p2
    cdef public np.ndarray p1
    cdef public np.ndarray p2
    def __init__(self, point_1, point_2):
        self.p1 = point_1  # treated as origin of the line
        self.p2 = point_2

    def subtended_angle(self, pos):
        return subtended_angle_c(pos, self.p1, self.p2)

    def distance_in_orientation(self, np.ndarray[np.float64_t, ndim=1] pos, double orientation):
        """Compute the distance from a point to the boundary, in a given angle.
        1. find equation for line from point and angle
        2. Find intersection point
        3. calculate distance between point (x,y) and intersection point.
        """
        cdef int valid
        cdef double ix, iy, r, s
        cdef double slope = tan(orientation)
        cdef np.ndarray point2 = pos + np.array([1, slope])
        ix, iy, valid, r, s = intersect_lines(pos, point2, self.p1, self.p2)

        cdef np.ndarray difvec = np.array([ix, iy]) - pos

        return sqrt(difvec[0] ** 2 + difvec[1] ** 2)


cdef class BVC(object):
    """Boundary Vector Cell. Parameters taken from Burgess et al.
    """
    cdef double beta   # mm
    cdef double sigma_0   # mm
    cdef double sigma_ang # radians
    cdef public double pref_distance, pref_orientation, sigma_rad

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

    def distance_to_nearest_boundary(self, np.ndarray[np.float64_t, ndim=1] pos, double orientation, env):
        cdef int idx = self.which_boundary(pos, orientation, env)
        b = env.boundaries[idx]
        cdef double d = b.distance_in_orientation(pos, orientation)
        cdef double a = b.subtended_angle(pos)
        return d, a

    def which_boundary(self,  np.ndarray[np.float64_t, ndim=1] pos, double orientation, env):
        cdef int i
        cdef np.ndarray v1
        cdef np.ndarray v2
        cdef double a1, a2
        for i in range(env.n_boundaries):
            b = env.boundaries[i]
            v1 = b.p1 - pos
            v2 = b.p2 - pos
            a1 = atan2(v1[1], v1[0]) % (2 * M_PI)
            a2 = atan2(v2[1], v2[0]) % (2 * M_PI)
            if in_smallest_interval(orientation, a1, a2):
                return i
        return False

    def compute_activation_pixel(self, np.ndarray[np.float64_t, ndim=1] pos, env):
        cdef double d, subtended_angle, theta, f
        cdef np.ndarray[np.float64_t, ndim=1] angles = np.linspace(0, 2 * np.pi, 400)[:-1]
        cdef int n_angles = len(angles)
        cdef int i
        cdef np.ndarray ds = np.empty(len(angles), dtype=np.float64)

        for i in range(n_angles):
            theta = angles[i]
            # get distance and subtended angle
            d, subtended_angle = self.distance_to_nearest_boundary(pos, theta, env)
            f = self.calculate_activation(d, subtended_angle, theta)
            ds[i] = f
        return ds.sum()

    def compute_ratemap(self, xs, ys, env):
        cdef int i, j
        cdef int nx = len(xs)
        cdef int ny = len(ys)
        cdef np.ndarray[np.float64_t, ndim=2] rate_map = np.zeros((len(xs), len(ys)), dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=1] pos

        for i in range(nx):
            for j in range(ny):
                pos = np.array([xs[i], ys[j]], dtype=np.float64)
                activation = self.compute_activation_pixel(pos, env)
                rate_map[i, j] = activation
        return rate_map

    def compute_ratemap_grid(self, xs, ys, env):
        cdef int i, j
        cdef int nx = len(xs)
        cdef int ny = len(ys)
        cdef np.ndarray[np.float64_t, ndim=1] rate_map = np.zeros(nx, dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=1] pos

        for i, j in zip(range(nx), range(ny)):
            pos = np.array([xs[i], ys[j]], dtype=np.float64)
            activation = self.compute_activation_pixel(pos, env)
            rate_map[i] = activation
        return rate_map

    def calculate_activation(self, double d, double subtended_angle, double theta):
        cdef double distance_term, angle_term, f
        # calculate activation
        distance_term = exp(-(d - self.pref_distance) ** 2 / (2 * self.sigma_rad ** 2)) / sqrt(
            2 * M_PI * self.sigma_rad ** 2)
        angle_term = exp(-(theta - self.pref_orientation) ** 2 / (2 * self.sigma_ang ** 2)) / sqrt(
            2 * M_PI * self.sigma_ang ** 2)
        f = distance_term * angle_term * subtended_angle
        return f




if __name__ == '__main__':


    b_left = ((0, 0), (0, 1000))
    b_top = ((0, 1000), (1000, 1000))
    b_right = ((1000, 0), (1000, 1000))
    b_bottom = ((0, 0), (1000, 0))

    b = Boundary(np.array(0,0), np.array(0,1000))


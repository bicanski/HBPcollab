import numpy as np
import math
from scipy.spatial import distance


def define_circle(xs, ys, centre, r):
    """
    Make a circle of ones.

    :param ndarray xs:
    :param ndarray ys:
    :param ndarray centre: x and y coordinate of the centre
    :param float r: Radius of the circle
    :return:
    """
    return (xs - centre[0]) ** 2 + (ys - centre[1]) ** 2 < r ** 2


def slice_number(x, y, n_slices=4):
    """Make radial, spoke like slices, and return the index of the slice that the coordinate (x,y) is in.

    :param x: x coordinate (can be an array)
    :param y: y coordinate (can be an array)
    :param n_slices: The number of slices
    :return: Index number of the slice that point (x,y) is in.
    """
    slice_idx = np.int32((np.pi + np.arctan2(y, x)) * (n_slices / (2*np.pi)))
    return slice_idx


def find_closest_vector(vec, arr):
    """Return the index of the vector in arr that has the smallest cosine distance to the input vector.

    :param vec: Vector to compare to an array of vectors
    :param arr: Array of vectors.
    :return int: The index of the array that is closest
    """
    distances = [distance.cosine(vec, arr[idx]) for idx in arr]
    return int(np.argmin(distances))


def gauss2d(position, var, centre):
    """
    Define a 2-D Gaussian with variance var and mean centre.

    :param position:
    :param var: Variance of the distribution
    :param centre: Mean of the distribution
    :return:
    """
    covariance_matrix = np.eye(2)*var

    exponent = -((position[0] - centre[0]) ** 2 + (position[1] - centre[1]) ** 2) / (2 * var)
    denominator = np.sqrt((2 * np.pi)**2 * np.linalg.det(covariance_matrix))
    result = np.exp(exponent) / denominator
    return result


def rotation_matrix_2d(angle):
    """

    :param angle: In radians
    :return:
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.matrix([[c, -s], [s, c]])


def circular_mean(weights, angles):
    x = y = 0.
    for angle, weight in zip(angles, weights):
        x += math.cos(math.radians(angle)) * weight
        y += math.sin(math.radians(angle)) * weight

    mean = math.degrees(math.atan2(y, x))
    return mean


def circle_intersection(circle1, circle2):
    """Calculate the intersection points of two circles.

    :param circle1: tuple(x,y,radius)
    :param circle2: tuple(x,y,radius)
    :return: Tuple of intersection points (which are (x,y) tuple)
    """

    x1, y1, r1 = circle1
    x2, y2, r2 = circle2

    dx, dy = x2-x1, y2-y1
    distance = np.sqrt(dx*dx+dy*dy)
    if distance > r1+r2:
        return None  # No solutions, the circles are separate
    if distance < abs(r1-r2):
        return [(x2, y2), (x2,y2)]  # No solutions because one circle is contained within the other: output centre 2
    if distance == 0 and r1 == r2:
        return None  # Circles are coincident and there are an infinite number of solutions

    a = (r1**2 - r2**2 + distance**2) / (2*distance)  # from circle 1 centre to line through intersection points
    h = np.sqrt(r1*r1-a*a)
    xm = x1 + a*dx/distance
    ym = y1 + a*dy/distance
    xs1 = xm + h*dy/distance
    xs2 = xm - h*dy/distance
    ys1 = ym - h*dx/distance
    ys2 = ym + h*dx/distance

    return (xs1, ys1), (xs2, ys2)


def round_number(input, round_to):
    """Round a number to the nearest factor.

    :param input:
    :param round_to:
    :return:
    """
    return int(np.round(input / round_to)) * round_to


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def normalize_angle(angle, min, max):
    """Normalize angles to lie in a range.

    :param angle:
    :param min:
    :param max:
    """
    if angle <= -180:
        angle += 360
    elif angle > 180:
        angle -= 360
    return angle


def random_argmax(x):
    """Argmax operation, but if there are multiple maxima, return one randomly.

    :param x: Input data
    :return: chosen index
    """
    x = np.array(x)
    arg_maxes = (x == x.max())
    b = np.flatnonzero(arg_maxes)
    choice = np.random.choice(b)
    return choice


def all_argmax(x):
    """Argmax operation, but if there are multiple maxima, return all.

    :param x: Input data
    :return: chosen index
    """
    x = np.array(x)
    arg_maxes = (x == x.max())
    indices = np.flatnonzero(arg_maxes)
    return indices



def random_point_in_circle(centre, radius):
    """Generate a (uniform) random point in a circle with given centre and radius.

    Method: using polar coordinates, first draw angle theta from a uniform distribution between 0 and 2 pi.
    Then, draw the radius. Uniform will have too many points close to the circle. We want pdf_r(r)=(2/R^2) * r
    cumulative pdf_r is F_r = (2/R^2)* (r^2)/2, inverse cumulative pdf is r = R*sqrt(F_r)

    :param centre:
    :param radius:
    :return:
    """
    theta = np.random.rand() * 2 * np.pi
    r = radius * np.sqrt(np.random.rand())

    cartesian_coordinate = [r * np.cos(theta), r * np.sin(theta)]
    return np.array(cartesian_coordinate) + np.array(centre)


def exponentiate(M, exp):
    """Exponentiate a matrix element-wise. For a diagonal matrix, this is equivalent to matrix exponentiation.

    :param M:
    :param exp:
    :return:
    """
    num_rows = len(M)
    num_cols = len(M[0])
    exp_m = np.zeros((num_rows, num_cols))

    for i in range(num_rows):
        for j in range(num_cols):
            if M[i][j] != 0:
                exp_m[i][j] = M[i][j] ** exp

    return exp_m

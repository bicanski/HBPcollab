import numpy as np
import math
from scipy.spatial import distance
from scipy.spatial.ckdtree import cKDTree


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
    slice_idx = np.int32((np.pi + np.arctan2(y, x)) * (n_slices / (2 * np.pi)))
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
    covariance_matrix = np.eye(2) * var

    exponent = -((position[0] - centre[0]) ** 2 + (position[1] - centre[1]) ** 2) / (2 * var)
    denominator = np.sqrt((2 * np.pi) ** 2 * np.linalg.det(covariance_matrix))
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

    dx, dy = x2 - x1, y2 - y1
    distance = np.sqrt(dx * dx + dy * dy)
    if distance > r1 + r2:
        return None  # No solutions, the circles are separate
    if distance < abs(r1 - r2):
        return [(x2, y2), (x2, y2)]  # No solutions because one circle is contained within the other: output centre 2
    if distance == 0 and r1 == r2:
        return None  # Circles are coincident and there are an infinite number of solutions

    a = (r1 ** 2 - r2 ** 2 + distance ** 2) / (2 * distance)  # from circle 1 centre to line through intersection points
    h = np.sqrt(r1 * r1 - a * a)
    xm = x1 + a * dx / distance
    ym = y1 + a * dy / distance
    xs1 = xm + h * dy / distance
    xs2 = xm - h * dy / distance
    ys1 = ym - h * dx / distance
    ys2 = ym + h * dx / distance

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


def get_relative_angle(angle_1, angle_2):
    """Return the smallest difference in angles between two angles (in degrees).
    """
    a = angle_1 - angle_2
    a = (a + 180) % 360 - 180
    return a


def in_rectangle(point, shape, position):
    """Check if point is in a rectangle.

    :param (tuple) point: Point to be checked.
    :param (tuple) shape: Width and height of the rectangle.
    :param (tuple) position: Position of bottom left corner of the rectangle.
    :return:
    """
    within_width = np.logical_and(position[0] <= point[0], point[0] <= position[0] + shape[0])
    within_height = np.logical_and(position[1] <= point[1], point[1] <= position[1] + shape[1])
    return np.logical_and(within_width, within_height)


def zip_lists(list_1, list_2):
    """Appends two list with alternating elements from list 1 and list 2.

    Note: this method assumes lists of equal length. If one list is longer than the other list, the later elements of
    the list are ignored, and the resulting  list will have length len(shortest_list) * 2.

    :param list_1:
    :param list_2:
    :return:
    """
    if len(list_1) != len(list_2):
        print('Warning: lists are of unequal length. Result length will match (2x) shortest.')
    return [j for i in zip(list_1, list_2) for j in i]


def make_symmetric(mat):
    return np.maximum(mat, mat.transpose())


def softmax(x, beta=2):
    """Compute the softmax function.

    :param x: Data
    :param beta: Inverse temperature parameter.
    :return:
    """
    x = np.array(x)
    return np.exp(beta * x) / sum(np.exp(beta * x))


def find_closest_in_grid(x_grid, y_grid, points_to_check):
    """Search in grid to find point closest to input.

    :param x_grid:
    :param y_grid:
    :param points_to_check:
    :return:
    """
    rf_centres = np.dstack([x_grid.ravel(), y_grid.ravel()])[0]
    points_array = np.array(points_to_check)
    dist, indexes = do_kdtree(rf_centres, points_array)
    return indexes


def do_kdtree(grid, points):
    """Use a KD Tree to find the location in grid that lies closest to points.

    :param grid:
    :param points:
    :return:
    """
    mytree = cKDTree(grid)
    dist, indexes = mytree.query(points)
    return dist, indexes


def in_circle(point, centre, radius):
    return np.linalg.norm(point - centre) < radius


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

import math
import numpy as np
cimport numpy as np
from libc.math cimport sqrt, acos, M_PI
from libcpp cimport bool


def intersect_lines(np.ndarray[np.float64_t, ndim=1] pt1, np.ndarray[np.float64_t, ndim=1] pt2,
                    np.ndarray[np.float64_t, ndim=1] ptA, np.ndarray[np.float64_t, ndim=1] ptB):
    """this returns the intersection of Line(pt1,pt2) and Line(ptA,ptB)

    returns a tuple: (xi, yi, valid, r, s), where
    (xi, yi) is the intersection
    r is the scalar multiple such that (xi,yi) = pt1 + r*(pt2-pt1)
    s is the scalar multiple such that (xi,yi) = pt1 + s*(ptB-ptA)
        valid == 0 if there are 0 or inf. intersections (invalid)
        valid == 1 if it has a unique intersection ON the segment

    # the first line is pt1 + r*(pt2-pt1)
    # the second line is ptA + s*(ptB-ptA)

    # we need to find the (typically unique) values of r and s
    # that will satisfy
    #
    # (x1, y1) + r(dx1, dy1) = (x, y) + s(dx, dy)
    #
    # which is the same as
    #
    #    [ dx1  -dx ][ r ] = [ x-x1 ]
    #    [ dy1  -dy ][ s ] = [ y-y1 ]
    #
    # whose solution is
    #
    #    [ r ] = _1_  [  -dy   dx ] [ x-x1 ]
    #    [ s ] = determinant  [ -dy1  dx1 ] [ y-y1 ]
    #
    # where determinant = (-dx1 * dy + dy1 * dx)
    #
    # if determinant is too small, they're parallel
    #

    :param pt1:
    :param pt2:
    :param ptA:
    :param ptB:
    :return:
    """
    cdef double det_tolerance = 0.00000001
    cdef double x1, y1, x2, y2, dx1, dy1, x, y, xB, yB, dx, dy, determinant, det_inv

    # in component form:
    x1 = pt1[0]
    y1 = pt1[1]
    x2 = pt2[0]
    y2 = pt2[1]
    dx1 = x2 - x1
    dy1 = y2 - y1

    x = ptA[0]
    y = ptA[1]

    xB = ptB[0]
    yB = ptB[1]

    dx = xB - x
    dy = yB - y

    determinant = (-dx1 * dy + dy1 * dx)

    if math.fabs(determinant) < det_tolerance:
        return 0, 0, 0, 0, 0

    # now, the determinant should be OK
    det_inv = 1.0 / determinant

    # find the scalar amount along the "self" segment
    cdef double r = det_inv * (-dy * (x - x1) + dx * (y - y1))

    # find the scalar amount along the input line
    cdef double s = det_inv * (-dy1 * (x - x1) + dx1 * (y - y1))

    # return the average of the two descriptions
    cdef double xi = (x1 + r * dx1 + x + s * dx) / 2.0
    cdef double yi = (y1 + r * dy1 + y + s * dy) / 2.0
    return xi, yi, 1, r, s


def subtended_angle_numpy(np.ndarray[np.float64_t, ndim=1] pos,
                          np.ndarray[np.float64_t, ndim=1] p1,
                          np.ndarray[np.float64_t, ndim=1] p2):
    """Compute the subtended angle of the line segment p1-p2 to position pos.

    :param pos:
    :param p1:
    :param p2:
    :return:
    """
    cdef np.ndarray[np.float64_t, ndim=1] vec2, vec1
    vec2 = p2 - pos
    vec1 = p1 - pos
    cdef float angle = np.arccos(vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
    return angle


def subtended_angle_c(np.ndarray[np.float64_t, ndim=1] pos,
                      np.ndarray[np.float64_t, ndim=1] p1,
                      np.ndarray[np.float64_t, ndim=1] p2):
    """Compute the subtended angle of the line segment p1-p2 to position pos.

    :param pos:
    :param p1:
    :param p2:
    :return:
    """
    cdef np.ndarray[np.float64_t, ndim=1] vec2, vec1
    cdef double norm1, norm2
    vec2 = p2 - pos
    vec1 = p1 - pos
    norm1 = sqrt(vec1[0] ** 2 + vec1[1] ** 2)
    norm2 = sqrt(vec2[0] ** 2 + vec2[1] ** 2)
    dotproduct = vec1[0] * vec2[0] + vec1[1] * vec2[1]
    cdef double angle = acos(dotproduct / (norm1 * norm2))
    return angle


def in_smallest_interval(double n, double a1, double a2):
    """Check of angle n is in the smallest interval between a1 and a2.

    :param n:
    :param a1:
    :param a2:
    :return:
    """
    if a1 == a2:
        return False
    cdef double rel_angle = get_relative_angle(a1, a2)
    if rel_angle < 0:
        return angle_between(n, a1, a2)
    else:
        return angle_between(n, a2, a1)


def get_relative_angle(double angle_1, double angle_2):
    """Return the smallest difference in angles between two angles (in radians).
    """
    cdef double a = angle_1 - angle_2
    a = (a + M_PI) % (M_PI*2) - M_PI
    return a


def angle_between(n, a, b):
    """Attention: radians!
    """
    n = (M_PI * 2 + (n % (M_PI * 2))) % (M_PI * 2)
    a = (M_PI * 20000 + a) % (M_PI*2)
    b = (M_PI * 20000 + b) % (M_PI*2)

    if (a < b):
        return a <= n and n <= b
    else:
        return a <= n or n <= b



def intersect_lines_naive(pt1, pt2, ptA, ptB):
    """this returns the intersection of Line(pt1,pt2) and Line(ptA,ptB)

    returns a tuple: (xi, yi, valid, r, s), where
    (xi, yi) is the intersection
    r is the scalar multiple such that (xi,yi) = pt1 + r*(pt2-pt1)
    s is the scalar multiple such that (xi,yi) = pt1 + s*(ptB-ptA)
        valid == 0 if there are 0 or inf. intersections (invalid)
        valid == 1 if it has a unique intersection ON the segment

    :param pt1:
    :param pt2:
    :param ptA:
    :param ptB:
    :return:
    """
    det_tolerance = 0.00000001

    # the first line is pt1 + r*(pt2-pt1)
    # in component form:
    x1, y1 = pt1
    x2, y2 = pt2
    dx1 = x2 - x1
    dy1 = y2 - y1

    # the second line is ptA + s*(ptB-ptA)
    x, y = ptA
    xB, yB = ptB
    dx = xB - x
    dy = yB - y

    # we need to find the (typically unique) values of r and s
    # that will satisfy
    #
    # (x1, y1) + r(dx1, dy1) = (x, y) + s(dx, dy)
    #
    # which is the same as
    #
    #    [ dx1  -dx ][ r ] = [ x-x1 ]
    #    [ dy1  -dy ][ s ] = [ y-y1 ]
    #
    # whose solution is
    #
    #    [ r ] = _1_  [  -dy   dx ] [ x-x1 ]
    #    [ s ] = determinant  [ -dy1  dx1 ] [ y-y1 ]
    #
    # where determinant = (-dx1 * dy + dy1 * dx)
    #
    # if determinant is too small, they're parallel
    #
    determinant = (-dx1 * dy + dy1 * dx)

    if math.fabs(determinant) < det_tolerance:
        return 0, 0, 0, 0, 0

    # now, the determinant should be OK
    det_inv = 1.0 / determinant

    # find the scalar amount along the "self" segment
    r = det_inv * (-dy * (x - x1) + dx * (y - y1))

    # find the scalar amount along the input line
    s = det_inv * (-dy1 * (x - x1) + dx1 * (y - y1))

    # return the average of the two descriptions
    xi = (x1 + r * dx1 + x + s * dx) / 2.0
    yi = (y1 + r * dy1 + y + s * dy) / 2.0
    return xi, yi, 1, r, s
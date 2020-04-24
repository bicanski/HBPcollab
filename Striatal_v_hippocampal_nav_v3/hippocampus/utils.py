import numpy as np
import ast


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


def softmax(x, beta=2):
    """Compute the softmax function.
    :param x: Data
    :param beta: Inverse temperature parameter.
    :return:
    """
    x = np.array(x)
    return np.exp(beta * x) / sum(np.exp(beta * x))


def get_relative_angle(angle_1, angle_2):
    """Return the smallest difference in angles between two angles (in degrees).
    """
    a = angle_1 - angle_2
    a = (a + 180) % 360 - 180
    return a


def angle_to_landmark(agent_location, landmark_centre, agent_orientation):
    """Get the relative direction to the landmark from the viewpoint of the

    :return:
    """
    relative_cue_pos = to_agent_frame(landmark_centre, agent_location, agent_orientation)
    angle = np.arctan2(relative_cue_pos[1], relative_cue_pos[0])
    return np.degrees(angle)


def to_agent_frame(object_location, agent_location, agent_direction):
    """Shift reference frame to agent's current location and direction.

    :param object_location:
    :param agent_location:
    :param agent_direction:
    :return:
    """
    translate = np.array(object_location) - np.array(agent_location)
    rotation_mat = rotation_matrix_2d(agent_direction).T
    result = rotation_mat.dot(translate)
    return np.asarray(result).squeeze()


def rotation_matrix_2d(angle):
    """

    :param angle: In radians
    :return:
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def from_np_array(array_string):
    array_string = ','.join(array_string.replace('[ ', '[').split())
    return np.array(ast.literal_eval(array_string))

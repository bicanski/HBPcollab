import numpy as np
from scipy.spatial.ckdtree import cKDTree

import utils


def to_agent_reference_frame(object_location, agent_location, agent_direction):
    """Shift reference frame to agent's current location and direction.

    :param object_location:
    :param agent_location:
    :param agent_direction:
    :return:
    """
    translate = np.array(object_location) - np.array(agent_location)
    rotation_mat = utils.rotation_matrix_2d(agent_direction).T
    result = rotation_mat.dot(translate)
    return np.asarray(result).squeeze()


def to_ext_reference_frame(object_location, agent_location, agent_direction):
    """Shift reference frame from agent's to world reference frame.

    :param object_location:
    :param agent_location:
    :param agent_direction:
    :return:
    """
    rotation_mat = utils.rotation_matrix_2d(agent_direction)
    rotated = rotation_mat.dot(object_location)
    rotated = np.asarray(rotated).squeeze()
    translate = np.add(rotated.T, np.array(agent_location))
    return translate


def make_receptive_fields(viewing_range=1, n_angles=20):
    """Give receptive field location using polar coordinates and

    :return:
    """
    # Make polar coordinates with a meshgrid.
    radii = np.logspace(.1, 1, 5) / 10 * viewing_range
    angles = np.linspace(-.75 * np.pi, .75 * np.pi, n_angles)
    radius_matrix, theta_matrix = np.meshgrid(radii, angles)
    # Conversion to cartesian coordinates
    x_grid = radius_matrix * np.cos(theta_matrix)
    y_grid = radius_matrix * np.sin(theta_matrix)
    return x_grid, y_grid


def make_receptive_fields_simple(n_angles=3, radius=2, n_radii=5, max_angle=135):
    """Give receptive field location using polar coordinates and

    :return:
    """
    max_angle = np.radians(max_angle)
    # Make polar coordinates with a meshgrid.
    # radii = radius
    radii = np.logspace(.1, 1, n_radii) / 10 * radius
    if n_radii == 1:
        radii = radius
    angles = np.linspace(-max_angle, max_angle, n_angles)
    radius_matrix, theta_matrix = np.meshgrid(radii, angles)
    # Conversion to cartesian coordinates
    x_grid = radius_matrix * np.cos(theta_matrix)
    y_grid = radius_matrix * np.sin(theta_matrix)
    return x_grid, y_grid


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


def gaussian_response(x_grid, y_grid, stimulus_location):
    """Compute response of a grid of gaussian receptive fields to a stimulus location

    :param x_grid:
    :param y_grid:
    :param stimulus_location:
    :return:
    """
    rf_centres = np.dstack([x_grid.ravel(), y_grid.ravel()])[0]
    responses = utils.gauss2d(stimulus_location, .05, rf_centres.T)
    return responses

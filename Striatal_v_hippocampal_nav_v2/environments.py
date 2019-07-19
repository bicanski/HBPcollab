import numpy as np
import utils
from matplotlib import patches
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import manifold
import os


class WaterMazeEnv(object):
    """Class for the morris water maze environment, specifying the possible actions, the location of the platform,
    and other relevant variables.
    """

    def __init__(self):
        self.time_bin = .1  # in seconds
        self.timeout = 120
        self.t = 0
        self.total_time = 0
        self.trial = 0
        self.reward = 1

        # Current position of the agent
        self.start_x = 1  # 0
        self.start_y = 1  # 0
        self.curr_x = self.start_x
        self.curr_y = self.start_y
        self.curr_orientation = 0
        self.curr_action_idx = 0

        # Initialise environment settings
        self.swimming_speed = .3 * self.time_bin  # in m / time step
        self.minx = 0
        self.maxx = 2
        self.miny = 0
        self.maxy = 2
        self.maze_centre = np.array([1, 1])  # np.array([0, 0])
        self.maze_radius = 1

        xs = np.linspace(self.minx, self.maxx, 1000)
        ys = np.linspace(self.miny, self.maxy, 1000)
        self.xs, self.ys = np.meshgrid(xs, ys)
        self.platform_centre = self.maze_centre + np.array([0.5, 0.2])
        self.platform_radius = .05

        self.landmark_1_centre = self.platform_centre + np.array([0, 0 + .1])
        self.landmark_1_radius = .025
        self.landmark_2_centre = self.platform_centre + np.array([-.1, .2])
        self.landmark_locations = [self.landmark_1_centre]

        # The possible actions in the watermaze are moving directions, with 0 angle being east, as in the unit circle
        self.movement_precision = 15
        self.max_turning_angle = 60
        self.actions = np.arange(-self.max_turning_angle, self.max_turning_angle + 1, self.movement_precision)
        self.all_orientations = np.arange(0, 360, self.movement_precision)
        self.reset()

    def set_platform_and_landmark(self, platform_location, landmark_location):
        self.platform_centre = platform_location
        self.landmark_1_centre = landmark_location
        self.landmark_locations = [self.landmark_1_centre]

    def get_action(self, action_id):
        return self.actions[action_id]

    def within_maze(self, position):
        """Create a Boolean mask to check whether a position is within the maze or not.

        :param (np.array) position: (x,y) position to be checked.
        :return (bool): True if agent on platform, False otherwise.
        """
        distance = np.linalg.norm(np.array(position).T - self.maze_centre, axis=1)
        return distance < self.maze_radius

    def reset(self):
        self.curr_x = self.start_x
        self.curr_y = self.start_y
        self.curr_orientation = np.random.choice(self.all_orientations)

    def compute_new_position(self, direction):
        next_x = (self.curr_x + self.swimming_speed * np.cos(np.radians(direction)))
        next_y = (self.curr_y + self.swimming_speed * np.sin(np.radians(direction)))
        return next_x, next_y

    def act(self, action):
        """

        :param action:
        :return:
        """
        self.curr_action_idx = np.flatnonzero(self.actions == action)[0]
        new_direction = (self.curr_orientation + action) % 360
        next_x, next_y = self.compute_new_position(new_direction)
        next_x, next_y, new_direction = self.prevent_boundary_collision(next_x, next_y, new_direction)

        self.curr_x = next_x
        self.curr_y = next_y
        self.curr_orientation = new_direction
        reached_platform = self.agent_at_goal()
        return reached_platform

    def prevent_boundary_collision(self, next_x, next_y, new_direction):
        """Check whether proposed position update hits the boundary and turn around if so.

        :param next_x:
        :param next_y:
        :param new_direction:
        :return:
        """
        dist_from_centre = np.linalg.norm(np.array([next_x, next_y]) - self.maze_centre)
        if dist_from_centre > self.maze_radius:
            new_direction = (self.curr_orientation + 180) % 360  # opposite direction
            next_x, next_y = self.compute_new_position(new_direction)
        return next_x, next_y, new_direction

    def agent_at_goal(self):
        return (self.curr_x - self.platform_centre[0]) ** 2 + \
               (self.curr_y - self.platform_centre[1]) ** 2 < self.platform_radius ** 2

    def random_location(self):
        """Give a (uniform) random location in the maze
        """
        x, y = utils.random_point_in_circle(self.maze_centre, self.maze_radius)  # self.initial_position
        return x, y


class PlusMaze(object):
    """Implementation of the plus shaped maze from the Packard and McGaugh (1996) experiment.
    """
    output_folder = './Data/PlusMaze'
    coordinates_filename = 'geodesic_coordinates.csv'

    def __init__(self):

        self.opposite_goal_box = [.9, 1]
        self.time_bin = .1  # in seconds
        self.timeout = 120
        self.t = 0
        self.total_time = 0
        self.trial = 0
        self.reward = 1

        # Current position of the agent
        self.start_x = .5
        self.start_y = .1
        self.start_orientation = 90
        self.curr_x = self.start_x
        self.curr_y = self.start_y
        self.curr_orientation = 0
        self.curr_action_idx = 0

        # Initialise environment settings
        self.swimming_speed = .7 * self.time_bin  # in m / time step
        self.minx = 0
        self.maxx = 1
        self.miny = 0
        self.maxy = 1
        self.maze_centre = np.array([.5, .5])  # np.array([0, 0])
        self.maze_length = self.maxy - self.miny
        self.maze_width = self.maxx - self.minx

        xs = np.linspace(self.minx, self.maxx, 100)
        ys = np.linspace(self.miny, self.maxy, 100)
        self.xs, self.ys = np.meshgrid(xs, ys)

        subsample_points = np.array([self.xs.flatten(), self.ys.flatten()])
        self.euclidean_coordinates = pd.DataFrame(subsample_points.T[self.within_maze(subsample_points)],
                                                  columns=['X', 'Y'])
        self.geodesic_coordinates = self.get_geodesic_coordinates()

        self.landmark_1_centre = np.array([.5, .6])

        self.landmark_locations = [self.landmark_1_centre]

        # The possible actions in the watermaze are moving directions, with 0 angle being east, as in the unit circle
        self.movement_precision = 15
        self.max_turning_angle = 60
        self.actions = np.arange(-self.max_turning_angle, self.max_turning_angle + 1, self.movement_precision)
        self.all_orientations = np.arange(0, 360, self.movement_precision)
        self.reset()

    def within_maze(self, position):
        """Check whether a given position is within the maze.

        The maze has a plus shape so I just check that the point is within the total defined area, and then that it is
        not in either one of four squares in the corners.

        :param position:
        :return:
        """
        in_maze_area = utils.in_rectangle(position, (self.maze_width, self.maze_length), (self.minx, self.miny))
        if not np.any(in_maze_area):
            return in_maze_area

        out_of_plus_maze = np.logical_or(
            np.logical_or(
                np.logical_or(
                    utils.in_rectangle(position, (.35, .35), (0, 0)),
                    utils.in_rectangle(position, (.35, .35), (0, .65))),
                utils.in_rectangle(position, (.35, .35), (.65, 0))),
            utils.in_rectangle(position, (.35, .35), (.65, .65)))

        return np.logical_and(in_maze_area, np.logical_not(out_of_plus_maze))

    def get_geodesic_coordinates(self):
        try:
            geo_coords = pd.DataFrame.from_csv(os.path.join(self.output_folder, self.coordinates_filename))
        except FileNotFoundError:
            print('\nNo coordinates file was found. Computing geodesic coordinates (this can take a few minutes)...\n')
            geo_coords = self.compute_geodesic_coordinates(self.euclidean_coordinates)
            geo_coords.to_csv(os.path.join(self.output_folder, self.coordinates_filename))
        return geo_coords

    def compute_geodesic_coordinates(self, euclid_coords):
        """Use the isomap algorithm (Tenenbaum et al., 2000, Science) to convert euclidean coordinates to geodesic.

        :return:
        """
        iso = manifold.Isomap(n_neighbors=4, path_method='FW')
        iso.fit(euclid_coords)
        geo_coords = iso.transform(euclid_coords)
        geo_coords = pd.DataFrame(geo_coords, columns=['GeoX', 'GeoY'])
        return geo_coords

    def lookup_geodesic_coordinate(self, euclidean_coordinates):
        """Find geodesic coordinate given Euclidean coordinate.

        The function first uses a KDTree to find the nearest neighbour in a set of Euclidean coordinates, and then uses
        a lookup table to find the corresponding geodesic coordinate.

        :param coordinate:
        :return:
        """
        distance, idx = utils.do_kdtree(np.array(self.euclidean_coordinates), np.array(euclidean_coordinates))
        geodesic_coordinate = self.geodesic_coordinates.iloc[idx]
        return np.array(geodesic_coordinate)

    def reset(self):
        self.curr_x = self.start_x
        self.curr_y = self.start_y
        self.curr_orientation = self.start_orientation

    def compute_new_position(self, direction):
        next_x = (self.curr_x + self.swimming_speed * np.cos(np.radians(direction)))
        next_y = (self.curr_y + self.swimming_speed * np.sin(np.radians(direction)))
        return next_x, next_y

    def act(self, action):
        """

        :param action:
        :return:
        """
        self.curr_action_idx = np.flatnonzero(self.actions == action)[0]
        new_direction = (self.curr_orientation + action) % 360
        next_x, next_y = self.compute_new_position(new_direction)
        next_x, next_y, new_direction = self.prevent_boundary_collision(next_x, next_y, new_direction)

        self.curr_x = next_x
        self.curr_y = next_y
        self.curr_orientation = new_direction
        reached_goal = self.agent_at_goal()
        return reached_goal

    def prevent_boundary_collision(self, next_x, next_y, new_direction):
        """Check whether proposed position update hits the boundary and turn around if so.

        :param next_x:
        :param next_y:
        :param new_direction:
        :return:
        """
        next_dir_try = 30
        changed_direction = new_direction
        while not self.within_maze([next_x, next_y]):

            left_turn = (new_direction + next_dir_try) % 360
            right_turn = (new_direction - next_dir_try) % 360

            next_x_plus30, next_y_plus30 = self.compute_new_position(left_turn)
            next_x_min30, next_y_min30 = self.compute_new_position(right_turn)

            if self.within_maze([next_x_plus30, next_y_plus30]) and not self.within_maze([next_x_min30, next_y_min30]):
                changed_direction = left_turn
            elif not self.within_maze([next_x_plus30, next_y_plus30]) and self.within_maze([next_x_min30, next_y_min30]):
                changed_direction = right_turn
            elif self.within_maze([next_x_plus30, next_y_plus30]) and self.within_maze([next_x_min30, next_y_min30]):
                changed_direction = np.random.choice([left_turn, right_turn])

            next_dir_try += 30
            next_x, next_y = self.compute_new_position(changed_direction)
        new_direction = changed_direction
        return next_x, next_y, new_direction

    def agent_at_goal(self):
        return self.curr_x < .1

    def agent_at_terminus(self):
        return self.curr_x < .1 or self.curr_x > .9 or self.in_opposite_goal_box()

    def draw_maze(self, face_color='gray'):
        plt.gca().add_patch(patches.Rectangle((0, 0), .35, .35, facecolor=face_color))
        plt.gca().add_patch(patches.Rectangle((0, .65), .35, .35, facecolor=face_color))
        plt.gca().add_patch(patches.Rectangle((.65, 0), .35, .35, facecolor=face_color))
        plt.gca().add_patch(patches.Rectangle((.65, .65), .35, .35, facecolor=face_color))
        plt.axis('equal')

    def in_opposite_goal_box(self):
        if self.opposite_goal_box[0] <= self.curr_y <= self.opposite_goal_box[1]:
            return True
        else:
            return False

    def start_on_opposite_side(self):
        self.start_x = .5
        self.start_y = .9
        self.start_orientation = -90
        self.opposite_goal_box = [0, .1]
        self.landmark_1_centre = np.array([.5, .4])
        self.reward = 0

    def start_on_original_side(self):
        self.start_x = .5
        self.start_y = .1
        self.start_orientation = 90
        self.opposite_goal_box = [.9, 1]
        self.landmark_1_centre = np.array([.5, .6])
        self.reward = 1

import numpy as np


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
        self.reward = 0

        # Initialise environment settings
        self.swimming_speed = .3 * self.time_bin  # in m / time step

        xs = np.linspace(-1, 1, 1000)
        ys = np.linspace(-1, 1, 1000)
        self.xs, self.ys = np.meshgrid(xs, ys)

        self.platform_centre = [0.5, 0.5]
        self.platform_radius = .1#.05

        self.landmark = [0.5, 0, .05]

        # The possible actions in the watermaze are moving directions, with 0 angle being east, as in the unit circle
        self.n_possible_angles = 12
        turning_precision = 360 / self.n_possible_angles
        self.actions = {idx: direction for (idx, direction) in zip(range(self.n_possible_angles),
                                                                   np.arange(0, 360, turning_precision))}
        self.action_idx = range(self.n_possible_angles)

    def get_action(self, action_id):
        return self.actions[action_id]

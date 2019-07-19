import numpy as np


class Environment(object):
    def __init__(self, cues_present='A'):
        self.time_bin = .1
        self.timeout = 120

        self.xs = np.arange(5, 1010, 10)
        self.ys = np.arange(5, 1010, 10)
        self.minx = 0
        self.maxx = 1000
        self.miny = 0
        self.maxy = 1000

        self.platform_centre = np.array((750, 600))
        #self.platform_centre = np.array((800, 500))
        self.platform_radius = 50

        self.cue_A = np.array([750, 650])  # original
        self.cue_B = np.array([700, 650])

        self.cue_A = np.array([700, 650])
        self.cue_B = np.array([600, 600])

        self._cues_present = cues_present

        #self.cue_A = np.array([800, 380])
        #self.cue_B = np.array([800, 620])
        #self.cue_B = np.array([750, 700])  # original


        #self.cue_B = np.array([900, 550])
        self.objects = [np.array([750, 650]), np.array([750, 700])]
        self.cue_radius = 40

        self.reward = 1

        self.swimming_speed = 300 * self.time_bin  # in m / time step

        # Current position of the agent
        self.start_x = 500  # 0
        self.start_y = 500  # 0
        self.curr_x = self.start_x
        self.curr_y = self.start_y
        self.curr_orientation = 0
        self.curr_action_idx = 0

        # The possible actions in the watermaze are moving directions, with 0 angle being east, as in the unit circle
        self.movement_precision = 15
        self.max_turning_angle = 60
        self.actions = np.arange(-self.max_turning_angle, self.max_turning_angle + 1, self.movement_precision)
        self.all_orientations = np.arange(0, 360, self.movement_precision)
        self.reset()

    @property
    def cues_present(self):
        return self._cues_present

    @cues_present.setter
    def cues_present(self, value):
        assert value == 'A' or 'B' or 'both', 'value has to be A, B, or both'
        if value == 'A':
            self.objects = [self.cue_A]
        elif value == 'B':
            self.objects = [self.cue_B]
        elif value == 'both':
            self.objects = [self.cue_A, self.cue_B]
        self._cues_present = value

    def agent_at_goal(self):
        return (self.curr_x - self.platform_centre[0]) ** 2 + \
               (self.curr_y - self.platform_centre[1]) ** 2 < self.platform_radius ** 2

    def reset(self):
        self.curr_x = self.start_x
        self.curr_y = self.start_y
        self.curr_orientation = 0  # np.random.choice(self.all_orientations)

    def compute_new_position(self, direction):
        next_x = (self.curr_x + self.swimming_speed * np.cos(np.radians(direction)))
        next_y = (self.curr_y + self.swimming_speed * np.sin(np.radians(direction)))
        return next_x, next_y

    def prevent_boundary_collision(self, next_x, next_y, new_direction):
        if next_x < self.minx or next_x > self.maxx or next_y < self.miny or next_y > self.maxy:
            new_direction = (self.curr_orientation + 180) % 360  # opposite direction
            next_x, next_y = self.compute_new_position(new_direction)
        return next_x, next_y, new_direction

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

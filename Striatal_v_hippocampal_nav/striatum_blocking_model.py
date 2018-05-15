import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import utils
import sensory_system.visual_system as vs


class Agent(object):

    def __init__(self, learning_rate=.1, n_trials=20, ax=None):

        self.time_bin = .1
        self.gamma = .98
        self.learning_rate = learning_rate
        self.epsilon = 0.05
        self.n_trials = n_trials

        self.t = 0
        self.total_time = 0
        self.trial = 0
        self.reward = 0
        self.total_reward = 0
        self.time_out = 1200 # was 200

        self.linear_velocity = .3 * self.time_bin  # Was 1
        self.angular_velocity = .4
        self.current_direction = 0

        self.minx = 0
        self.maxx = 2
        self.miny = 0
        self.maxy = 2

        self.platform_centre = np.array([1.5, 1.2])
        self.platform_radius = .05  # .5

        self.landmark_1_centre = self.platform_centre + np.array([.1, -.15])
        self.landmark_1_radius = .025
        self.landmark_2_centre = self.platform_centre + np.array([-.1, .2])
        self.landmark_locations = [self.landmark_1_centre]

        self.maze_centre = [1, 1]
        self.maze_radius = 1

        angles = np.linspace(0, 2*np.pi, 20)
        x_marks = np.cos(angles) * self.maze_radius + self.maze_centre[0]
        y_marks = np.sin(angles) * self.maze_radius + self.maze_centre[1]
        self.border_marks = np.array([x_marks, y_marks]).T

        self.initial_position = self.random_location()
        self.x, self.y = self.initial_position

        self.locs = []
        self.locs.append([self.x, self.y])
        self.dirs = []
        self.dirs.append(self.current_direction)

        self.escape_times = []

        # sensory system:
        self.rf_radius = 1
        self.max_viewing_angle = 135
        self.rf_x, self.rf_y = vs.make_receptive_fields_simple(n_angles=21,
                                                               radius=self.rf_radius,
                                                               max_angle=self.max_viewing_angle,
                                                               n_radii=6)
        self.sens_neuron_activations = np.zeros(self.rf_x.flatten().shape)
        self.previous_sensory_activations = np.zeros(self.sens_neuron_activations.shape)
        self.max_firing_rate = 10
        self.rf_var = .05

        # RL system:
        self.movement_precision = 10  # smallest possible turning angle
        self.actions = np.radians(np.arange(-60, 61, self.movement_precision))  # directions of movement, 0 being straight ahead
        self.possible_directions = np.radians(np.arange(0, 360, self.movement_precision))
        self.current_action = 0
        self.striatum_activation = np.zeros(self.actions.shape)
        self.previous_striatum_activation = None
        self.weight_mat = np.zeros((self.striatum_activation.shape[0], self.sens_neuron_activations.shape[0]))
        self.delta = 0

        # Initialise position log:
        self.log = np.array([[self.x, self.y, self.trial]])
        self.position_log = None

        # Set up plotting parameters
        if ax is not None:
            self.line, = ax.plot([], [], 'o')
            self.hit, = ax.plot([], [], 'ro')
            self.rot_text = ax.text(0.05, 0.95, '', horizontalalignment='left', verticalalignment='top',
                                    transform=ax.transAxes)
            self.ax = ax
            self.ax.set_xlim(0, 10)
            self.ax.set_ylim(0, 10)
            self.ax.grid(True)
            landmark = plt.Circle(self.landmark_1_centre, self.landmark_1_radius, color='g')
            ax.add_artist(landmark)
            platform = plt.Circle(self.platform_centre, self.platform_radius, color='b')
            ax.add_artist(platform)

    def run_simulation(self):
        for trial in range(self.n_trials):
            reached_platform = False
            while not reached_platform and self.t < self.time_out:
                reached_platform = self.take_step()

            self.trial_reset()
        self.position_log = self.save_log()

    def run_blocking_experiment(self):
        for trial in range(self.n_trials):
            reached_platform = False

            if self.trial <= int(self.n_trials / 3):
                self.landmark_locations = [self.landmark_1_centre]
            elif int(self.n_trials / 3) < self.trial < int(self.n_trials * 2 / 3):
                self.landmark_locations = [self.landmark_1_centre, self.landmark_2_centre]
            elif self.trial >= int(self.n_trials * 2 / 3):
                self.landmark_locations = [self.landmark_2_centre]

            while not reached_platform and self.t < self.time_out:
                reached_platform = self.take_step()

            self.trial_reset()
        self.position_log = self.save_log()

    def take_step(self, action=None):
        self.t += 1
        self.total_time += 1
        self.update_position(action=action)
        self.update_sensory_neurons()
        self.striatum_activation = self.compute_striatum_activation()
        reached_platform = self.agent_on_platform()
        if reached_platform:
            self.reward = 1
        self.update_Q_weights()
        self.total_reward += self.reward
        self.log_positions()
        return reached_platform

    def update_position(self, action=None):
        new_direction = self.choose_direction(action)
        next_x, next_y = self.compute_new_position(new_direction)
        next_x, next_y, new_direction = self.boundary_collision(next_x, next_y, new_direction)

        self.current_direction = new_direction
        self.x, self.y = [next_x, next_y]

    def compute_new_position(self, new_direction):
        next_x = (self.x + self.linear_velocity * np.cos(new_direction))
        next_y = (self.y + self.linear_velocity * np.sin(new_direction))
        return next_x, next_y

    def choose_direction(self, action):
        if self.t % 1 == 0:
            self.current_action = self.e_greedy_selection()
            if action is not None:
                self.current_action = action
            new_direction = (self.current_direction + self.actions[self.current_action]) % (2 * np.pi)
        else:
            new_direction = self.current_direction
        return new_direction

    def boundary_collision(self, x, y, new_direction):
        """Check whether proposed position update hits the boundary and turn around if so.

        :param x:
        :param y:
        :param new_direction:
        :return:
        """
        dist_from_centre = np.linalg.norm(np.array([x, y]) - self.maze_centre)
        if dist_from_centre > self.maze_radius:
            new_direction = (self.current_direction + np.pi) % (2 * np.pi)
            x, y = self.compute_new_position(new_direction)
        return x, y, new_direction

    def trial_reset(self):
        """Reset the agent to its starting position.
        """
        # print('reached platform for the {}th time'.format(self.trial+1))
        self.escape_times.append(self.t)
        self.t = 0
        self.reward = 0
        self.delta = 0
        self.x, self.y = self.random_location()
        self.trial += 1
        self.current_direction = np.random.choice(self.possible_directions)
        self.update_sensory_neurons()
        self.striatum_activation = np.zeros(self.actions.shape)
        self.striatum_activation = self.compute_striatum_activation()

    def random_location(self):
        """Give a (uniform) random location in the maze

        :return:
        """
        x, y = utils.random_point_in_circle(self.maze_centre, self.maze_radius)  # self.initial_position
        return x, y

    def e_greedy_selection(self):
        """Epsilon greedy action selection: choose greedy action or random with probability epsilon.

        :return (int): Index of chosen action
        """
        if np.random.rand() < self.epsilon:
            chosen_action = np.random.choice(range(self.actions.__len__()))
        else:
            chosen_action = utils.random_argmax(self.striatum_activation)
        return chosen_action

    def update_sensory_neurons(self):
        """Translate and rotate view to agent's reference frame, then check angle smaller than max viewing angle, then
        update neurons with gaussian RFs.

        :return:
        """
        self.previous_sensory_activations = self.sens_neuron_activations
        self.sens_neuron_activations = np.zeros(self.rf_x.flatten().shape)
        self.detect_landmarks()
        self.detect_border()
        self.sens_neuron_activations = np.minimum(self.sens_neuron_activations, self.max_firing_rate)

    def detect_landmarks(self):
        for landmark, loc in enumerate(self.landmark_locations):
            if self.landmark_in_range(loc):
                rel_landmark_location = vs.to_agent_reference_frame(loc,
                                                                    [self.x, self.y],
                                                                    self.current_direction)
                relative_angle = np.arctan2(rel_landmark_location[1], rel_landmark_location[0])
                if abs(relative_angle) <= np.radians(self.max_viewing_angle):
                    # set rate between 0 and 10:
                    elicited_response = self.max_firing_rate * vs.gaussian_response(self.rf_x,
                                                                                    self.rf_y,
                                                                                    rel_landmark_location) / 3.81
                    self.sens_neuron_activations += elicited_response

    def detect_border(self):
        centre_relative = vs.to_agent_reference_frame(self.maze_centre,
                                                      [self.x, self.y],
                                                      self.current_direction)
        distance = np.linalg.norm(centre_relative)
        wall_visible = (distance < self.rf_radius + self.maze_radius) & \
                       (distance > abs(self.rf_radius - self.maze_radius))
        if wall_visible:
            rf_centres = np.dstack([self.rf_x.ravel(), self.rf_y.ravel()])[0]
            distance_to_centre = np.linalg.norm(np.add(rf_centres, -centre_relative), axis=1)
            denominator = (np.sqrt((2 * np.pi)**2 * np.linalg.det(np.eye(2)*self.rf_var)))
            elicited_response = np.exp((-distance_to_centre/(2*self.rf_var))) / denominator
            self.sens_neuron_activations += elicited_response

    def compute_striatum_activation(self):
        self.previous_striatum_activation = self.striatum_activation
        input_current = np.matmul(self.weight_mat, self.sens_neuron_activations)
        striatum_activation = self.neural_activation(input_current)
        return striatum_activation

    def update_Q_weights(self):
        """Use Q-learning rule to update weights.

        :return:
        """
        self.compute_prediction_error()
        delta_q = self.learning_rate * self.delta
        if sum(self.previous_sensory_activations) != 0:
            weight_increment = delta_q * self.previous_sensory_activations / sum(self.previous_sensory_activations)
            self.weight_mat[self.current_action] += weight_increment

    def compute_prediction_error(self):
        if self.reward == 1:
            self.delta = self.reward - self.previous_striatum_activation[self.current_action]
        else:
            self.delta = self.gamma * max(self.striatum_activation) - \
                         self.previous_striatum_activation[self.current_action]

    def agent_on_platform(self):
        """Check if agent's current position is on the platform.

        :return (bool): True if agent on platform, false otherwise.
        """
        return (self.x - self.platform_centre[0]) ** 2 + \
               (self.y - self.platform_centre[1]) ** 2 < self.platform_radius ** 2

    def landmark_in_range_dep(self, landmark_idx):
        distance_to_landmark = np.linalg.norm(np.array([self.x, self.y]) -
                                              np.array(self.landmark_locations[landmark_idx]))
        landmark_visible = distance_to_landmark < (self.landmark_1_radius + self.rf_radius)
        return landmark_visible

    def landmark_in_range(self, loc):
        distance_to_landmark = np.linalg.norm(np.array([self.x, self.y]) -
                                              np.array(loc))
        landmark_visible = distance_to_landmark < self.rf_radius
        return landmark_visible

    @staticmethod
    def neural_activation(x):
        return np.tanh(x)

    def log_positions(self):
        """Add current position to the position log.
        """
        self.log = np.append(self.log, np.array([[self.x, self.y, self.trial]]), axis=0)

    def save_log(self):
        """Return a log of the positions as a pandas dataframe.

        :return:
        """
        position_log = pd.DataFrame(self.log, columns=['X position', 'Y position', 'Trial'])
        position_log.index.name = 'Time bin'
        return position_log


a = Agent(n_trials = 30)
a.run_blocking_experiment()
import plotting as pl
pl.plot_trace(a, range(a.n_trials))

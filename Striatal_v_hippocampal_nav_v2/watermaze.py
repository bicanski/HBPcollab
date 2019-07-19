import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

from environments import WaterMazeEnv
import utils

plt.style.use('seaborn')


class WaterMazeAgent(object):
    """Base class for agents navigating the water maze containing general methods.
    """
    def __init__(self, n_trials, env=WaterMazeEnv()):

        self.env = env
        self.n_trials = n_trials

        # Initialise agent settings
        self.reward = 0
        self.total_reward = 0
        self.learning_rate = .1
        self.beta = 2  # exploration/exploitation index
        self.gamma = .9975  # future rewards discount factor
        self.delta = 0

        self.initial_position = [-.7, -.7]
        self.current_position = self.initial_position
        self.reached_platform = False
        self.current_action = None
        self.actions = None

        # Initialise position log:
        self.position_log = pd.DataFrame(columns=['X position', 'Y position', 'Trial'])
        self.position_log.index.name = 'Time bin'
        self.position_log.loc[0] = [self.current_position[0], self.current_position[1], self.env.trial]

    def get_position_update(self, action_idx):
        """Get the position update resulting from choosing a certain action.

        :param action_idx: Index of the action chosen.
        :return: Array with x and y position updates.
        """
        dx = self.env.swimming_speed * np.cos(np.radians(self.env.actions[action_idx]))
        dy = self.env.swimming_speed * np.sin(np.radians(self.env.actions[action_idx]))
        position_update = np.array([dx, dy])
        return position_update

    def apply_momentum(self, policy_sampled_action):
        """Apply momentum to the action sampled from the policy by taking the circular mean of that action and the
        previous heading direction in a ratio 1:3 (cf. Foster et al. 2000). This restricts the turning curve of the
        agent.

        :param policy_sampled_action:
        :return:
        """
        weights_previous_vs_policy = [3, 1]
        weighted_sum = utils.circular_mean(weights_previous_vs_policy,
                                           [self.env.actions[self.previous_action],
                                            self.env.actions[policy_sampled_action]])
        angle = utils.round_number(weighted_sum, 30) % 360
        resulting_action = np.where(np.arange(0, 360, 30) == angle)[0][0]
        return resulting_action

    @staticmethod
    def within_maze(position):
        """Create a Boolean mask to check whether a position is within the maze or not.

        :param (np.array) position: (x,y) position to be checked.
        :return (bool): True if agent on platform, false otherwise.
        """
        return position[0] ** 2 + position[1] ** 2 < 1 ** 2

    def on_platform(self):
        """Check if agent's current position is on the platform.

        :return (bool): True if agent on platform, false otherwise.
        """
        return (self.current_position[0] - self.env.platform_centre[0]) ** 2 + \
               (self.current_position[1] - self.env.platform_centre[1]) ** 2 < self.env.platform_radius ** 2

    def check_reward(self):
        return 1.0 * utils.define_circle(self.current_position[0], self.current_position[1], self.env.platform_centre,
                                                                                            self.env.platform_radius)

    def check_wall_collision(self, position_update):
        """Check whether the proposed position update would put the agent outside the maze walls, and if so change
        the direction.

        :param position_update: Array with x and y updates.
        """
        if not self.within_maze(self.current_position + position_update):
            left_possible = self.within_maze(self.current_position +
                                             self.get_position_update((self.current_action - 1) % len(self.actions)))
            # TODO: Make this method waterproof by iteratively checking possibility.
            right_possible = self.within_maze(self.current_position +
                                             self.get_position_update((self.current_action + 1) % len(self.actions)))
            if not left_possible and not right_possible:
                self.current_action = int((self.current_action + len(self.actions)/2) % len(self.actions))
            elif left_possible and not right_possible:
                self.current_action = (self.current_action - 2) % len(self.actions)
            elif not left_possible and right_possible:
                self.current_action = (self.current_action + 2) % len(self.actions)

    def log_positions(self):
        """Add current position to the position log.
        """
        self.position_log.loc[self.env.total_time] = \
            [self.current_position[0], self.current_position[1], self.env.trial]

    def plot_trial_times(self):
        trials = np.arange(self.n_trials)
        number_of_time_points = [self.position_log[self.position_log['Trial'] == i].shape[0] for i in trials]
        trial_time = np.array(number_of_time_points) * self.env.time_bin
        plt.plot(trials[1:], trial_time[1:])

    # Plotting functions:
    # TODO: move plotting functions outside of class. Consider plotting class?

    def draw_maze_and_platform(self, ax):
        """Draw the maze and the platform on the specified axis.

        :param ax: Matplotlib axis to draw on
        :return:
        """
        angles = np.linspace(0, 2 * np.pi, 100)
        ax.plot(np.cos(angles), np.sin(angles))
        platform = plt.Circle(self.env.platform_centre, self.env.platform_radius, color='g')
        ax.add_artist(platform)

    def plot_trace(self, trials_to_plot=None):
        """Plot the walking trajectory in the maze.
        """
        if not trials_to_plot:
            trials_to_plot = [1, int(self.n_trials/2)+1, self.n_trials]

        n_rows = int(math.ceil(len(trials_to_plot)/5))
        n_cols = int(math.ceil(len(trials_to_plot)/n_rows))
        fig, axs = plt.subplots(n_rows, n_cols, sharex='row', sharey='row')
        angles = np.linspace(0, 2 * np.pi, 100)

        axs = axs.ravel()

        for i, trial in enumerate(trials_to_plot):
            axs[i].plot(np.cos(angles), np.sin(angles))  # Draw the boundary of the circular maze
            trial_trajectory = self.position_log[self.position_log['Trial'] == trial]
            axs[i].plot(trial_trajectory['X position'], trial_trajectory['Y position'])
            axs[i].axis('equal')  # enforces equal axis sizes
            axs[i].set_title('Trial {}'.format(trial))

            platform = plt.Circle(self.env.platform_centre, self.env.platform_radius, color='g')
            axs[i].add_artist(platform)
            plt.xlim((-1, 1))
            plt.ylim((-1, 1))
            axs[i].tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off')


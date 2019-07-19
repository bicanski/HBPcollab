import numpy as np
from environments import WaterMazeEnv
from sensory_system import visual_system as vs
from utils import get_relative_angle
import utils


class TDStriatum(object):
    """This class implements the striatum Q learning model, learning incrementally through model-free RL.
    """
    def __init__(self, env=WaterMazeEnv(), learning_rate=.001, gamma=.98, lamb=.76):
        """

        :param env: Instance of environment class.
        :param learning_rate: Learning rate for Q learning.
        :param gamma: Future reward discount factor.
        :param lamb: Eligibility trace decay parameter.
        """
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lamb = lamb

        # Make the landmark cells
        self.field_width = 5  # 27.5
        self.max_viewing_angle = 175  # 135
        self.n_landmark_cells = 100
        self.landmark_cell_centres = np.linspace(-self.max_viewing_angle, self.max_viewing_angle, self.n_landmark_cells)
        self.LC_activations = np.zeros(self.n_landmark_cells)
        self.previous_LC_activity = None

        # Make the action cells
        self.n_action_cells = len(self.env.actions)
        self.previous_striatum_activation = None
        self.striatum_activation = np.zeros(self.n_action_cells)
        self.weight_mat = np.zeros((self.n_action_cells, self.n_landmark_cells))
        self.weight_mat = np.random.rand(self.n_action_cells, self.n_landmark_cells) * .004
        self.eligibility_trace = np.zeros(self.weight_mat.shape)
        self.generalisation_phase_activity = None
        self.generalisation_phase_var = 22.5

    def update(self):
        # Update all cell responses given new location
        reward = 0
        self.compute_gen_phase()
        self.update_trace()

        self.LC_activations = self.compute_LC_activity()
        self.striatum_activation = self.compute_striatum_activation()

        reached_goal = self.env.agent_at_goal()
        if reached_goal:
            reward = self.env.reward
        self.update_weights(reward)

    def compute_gen_phase(self):
        chosen_direction = self.env.actions[self.env.curr_action_idx]
        normaliser = (2 * self.generalisation_phase_var)
        self.generalisation_phase_activity = \
            np.exp(-abs(get_relative_angle(self.env.actions, chosen_direction)) / normaliser)

    def update_weights(self, reward):
        prediction_error = self.compute_prediction_error(reward)
        delta_w = self.learning_rate * prediction_error * self.eligibility_trace
        self.weight_mat += delta_w
        self.normalise_weights()

    def normalise_weights(self):
        striatum_response = max(self.compute_striatum_activation(set_previous=False))
        if striatum_response > 10:
            self.weight_mat = self.weight_mat / striatum_response

    def update_trace(self):
        weight_importance = np.outer(self.generalisation_phase_activity, self.LC_activations)
        self.eligibility_trace = weight_importance + self.lamb * self.eligibility_trace

    def compute_prediction_error(self, reward):
        if reward == 1:
            delta = reward - self.previous_striatum_activation[self.env.curr_action_idx]
        else:
            delta = self.gamma * max(self.striatum_activation) - \
                         self.previous_striatum_activation[self.env.curr_action_idx]
        return delta

    def compute_LC_activity(self, width=27.5):
        """Compute landmark cell activitiy.

        The activity is given by a gaussian around the preferred angular distance of each landmark cell, following
        Dolle et al. (2010). The width of the gaussian is inversely proportional to the euclidean distance to of the
        landmark to the agent.

        :param width: Width of the receptive fields.
        :return:
        """
        self.previous_LC_activity = self.LC_activations
        activity = np.zeros(self.LC_activations.shape)
        for landmark in self.env.landmark_locations:
            # First compute the relative angle to the landmark
            landmark_direction = self.angle_to_landmark(landmark)
            # Then the activity
            angular_distances = abs(get_relative_angle(landmark_direction, self.landmark_cell_centres))
            euclidian_distance = np.linalg.norm(landmark - [self.env.curr_x, self.env.curr_y])
            activity += (1/ (euclidian_distance+.5) ) * np.exp(- (angular_distances ** 2 / (2 * (self.field_width) ** 2)))
            activity /= np.linalg.norm(activity)
        return activity

    def compute_striatum_activation(self, set_previous=True):
        if set_previous:
            self.previous_striatum_activation = self.striatum_activation
        input_current = np.matmul(self.weight_mat, self.LC_activations)
        striatum_activation = input_current
        return striatum_activation

    def angle_to_landmark(self, landmark_centre):
        """Get the relative direction to the landmark from the viewpoint of the

        :return:
        """
        relative_cue_pos = vs.to_agent_reference_frame(landmark_centre,
                                                       [self.env.curr_x, self.env.curr_y],
                                                       np.radians(self.env.curr_orientation))
        angle = np.arctan2(relative_cue_pos[1], relative_cue_pos[0])
        return np.degrees(angle)

    @staticmethod
    def neural_activation(x):
        return np.tanh(x)

    def choose_action(self):
        """Choose action by taking max over action cells.
        """
        if len(self.striatum_activation) != len(self.env.actions):
            print('something is wrong')
        action_idx = utils.random_argmax(self.striatum_activation)
        value = self.striatum_activation[action_idx]
        return self.env.actions[action_idx], value * 5

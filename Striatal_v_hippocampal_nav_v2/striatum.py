import numpy as np
from environments import WaterMazeEnv
from sensory_system import visual_system as vs
import utils


class Striatum(object):
    """This class implements the striatum Q learning model, learning incrementally through model-free RL.
    """
    def __init__(self, env=WaterMazeEnv(), learning_rate=.1, gamma=.98):

        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma

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
        self.striatum_activation = np.zeros(self.env.actions.shape)
        self.previous_striatum_activation = None
        self.weight_mat = np.zeros((self.striatum_activation.shape[0], self.sens_neuron_activations.shape[0]))
        self.delta = 0

    def update(self):
        reward = 0
        # Update all cell responses given new location
        self.update_sensory_neurons()
        self.striatum_activation = self.compute_striatum_activation()
        reached_platform = self.env.agent_at_goal()
        if reached_platform:
            reward = 1
        self.update_Q_weights(reward)

    def choose_action(self):
        """Choose action by taking max over action cells.
        """
        action_idx = utils.random_argmax(self.striatum_activation)
        value = self.striatum_activation[action_idx]
        return self.env.actions[action_idx], value

    def compute_striatum_activation(self):
        self.previous_striatum_activation = self.striatum_activation
        input_current = np.matmul(self.weight_mat, self.sens_neuron_activations)
        striatum_activation = self.neural_activation(input_current)
        return striatum_activation

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

    def update_Q_weights(self, reward):
        """Use Q-learning rule to update weights.

        :return:
        """
        self.compute_prediction_error(reward)
        delta_q = self.learning_rate * self.delta
        if sum(self.previous_sensory_activations) != 0:
            weight_increment = delta_q * self.previous_sensory_activations / sum(self.previous_sensory_activations)
            self.weight_mat[self.env.curr_action_idx] += weight_increment

    def compute_prediction_error(self, reward):
        if reward == 1:
            self.delta = reward - self.previous_striatum_activation[self.env.curr_action_idx]
        else:
            self.delta = self.gamma * max(self.striatum_activation) - \
                         self.previous_striatum_activation[self.env.curr_action_idx]

    def detect_landmarks(self):
        for landmark, loc in enumerate(self.env.landmark_locations):
            if self.landmark_in_range(loc):
                rel_landmark_location = vs.to_agent_reference_frame(loc,
                                                                    [self.env.curr_x, self.env.curr_y],
                                                                    self.env.curr_orientation)
                relative_angle = np.arctan2(rel_landmark_location[1], rel_landmark_location[0])
                if abs(relative_angle) <= np.radians(self.max_viewing_angle):
                    # set rate between 0 and 10:
                    elicited_response = self.max_firing_rate * vs.gaussian_response(self.rf_x,
                                                                                    self.rf_y,
                                                                                    rel_landmark_location) / 3.81
                    self.sens_neuron_activations += elicited_response

    def detect_border(self):
        centre_relative = vs.to_agent_reference_frame(self.env.maze_centre,
                                                      [self.env.curr_x, self.env.curr_y],
                                                      self.env.curr_orientation)
        distance = np.linalg.norm(centre_relative)
        wall_visible = (distance < self.rf_radius + self.env.maze_radius) & \
                       (distance > abs(self.rf_radius - self.env.maze_radius))
        if wall_visible:
            rf_centres = np.dstack([self.rf_x.ravel(), self.rf_y.ravel()])[0]
            distance_to_centre = np.linalg.norm(np.add(rf_centres, -centre_relative), axis=1)
            denominator = (np.sqrt((2 * np.pi)**2 * np.linalg.det(np.eye(2)*self.rf_var)))
            elicited_response = np.exp((-distance_to_centre/(2*self.rf_var))) / denominator
            self.sens_neuron_activations += elicited_response

    @staticmethod
    def neural_activation(x):
        return np.tanh(x)

    def landmark_in_range(self, loc):
        distance_to_landmark = np.linalg.norm(np.array([self.env.curr_x, self.env.curr_y]) -
                                              np.array(loc))
        landmark_visible = distance_to_landmark < self.rf_radius
        return landmark_visible

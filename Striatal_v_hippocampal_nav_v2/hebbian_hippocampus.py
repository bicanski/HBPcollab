import numpy as np
from environments import WaterMazeEnv, PlusMaze
from cognitive_tasks import DeterministicTask
import utils
from utils import get_relative_angle, make_symmetric
import matplotlib.pyplot as plt
from scipy.spatial import distance
from mpl_toolkits.mplot3d import Axes3D


class TDHippocampus(object):
    """This class implements a model hippocampus that uses TD learning to learn a value function using geodesic place
    cell features as a basis set, following Gustafson & Daw (2011).
    """
    def __init__(self, env=WaterMazeEnv(), learning_rate=.001, negative_learning_rate=.001, gamma=.98, lamb=.67):
        self.env = env
        self.learning_rate = learning_rate
        self.negative_learning_rate = negative_learning_rate
        self.gamma = gamma
        self.lamb = lamb
        self.max_turning_angle = 60

        # Create the place cells:
        self.field_width = .3
        self.field_centres = self.create_place_cells()
        if isinstance(self.env, PlusMaze):
            self.geodesic_field_centres = self.env.lookup_geodesic_coordinate(self.field_centres.T).T
        self.n_place_cells = self.field_centres.shape[1]
        self.max_response = self.get_max_response()
        self.previous_place_cell_responses = None
        self.place_cell_responses = None
        self.update_place_cell_response()

        # And the goal cell
        self.value = 0
        self.previous_value = 0

        self.weights = np.zeros(self.place_cell_responses.shape)
        self.weights = np.random.rand(self.n_place_cells) * .04
        self.eligibility_trace = np.zeros(self.weights.shape)

        # Allocentric actions (0 is east)
        # Note that the only actions currently available to the agent will be -60 to 60 degrees away.
        self.allocentric_directions = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]

        self.remembered_goal_location = [0, 0]

    def create_place_cells(self):
        """Create a grid of evenly spaced place field centres within the water maze.
        """
        field_centres_x, field_centres_y = np.meshgrid(
            np.linspace(self.env.minx, self.env.maxx, 12), np.linspace(self.env.miny, self.env.maxy, 12))
        field_centres = np.array([field_centres_x.flatten(), field_centres_y.flatten()])
        field_centres = field_centres[:, self.env.within_maze(field_centres)]
        return field_centres

    def update_place_cell_response(self):
        """Compute the firing rate of a each place cell with Gaussian tuning curves, for the current position of the
        agent
        """
        self.previous_place_cell_responses = self.place_cell_responses
        if isinstance(self.env, PlusMaze):
            x, y = self.env.lookup_geodesic_coordinate([self.env.curr_x, self.env.curr_y])
            self.place_cell_responses = utils.gauss2d([x, y], var=self.field_width ** 2,
                                                      centre=self.geodesic_field_centres)
            self.place_cell_responses /= np.linalg.norm(self.place_cell_responses)
        else:
            self.place_cell_responses = utils.gauss2d([self.env.curr_x, self.env.curr_y], var=self.field_width ** 2,
                                                      centre=self.field_centres)
            self.place_cell_responses /= np.linalg.norm(self.place_cell_responses)

    def compute_value(self):
        self.previous_value = self.value
        self.value = np.dot(self.weights, self.place_cell_responses)

    def update(self):
        # Update all cell responses given new location

        reward = 0

        self.update_place_cell_response()
        self.compute_value()

        reached_goal = self.env.agent_at_goal()
        if reached_goal:
            reward = self.env.reward

        self.update_trace()
        self.update_weights(reward)

    def update_trace(self):
        self.eligibility_trace = self.lamb * self.eligibility_trace + self.previous_place_cell_responses

    def update_weights(self, reward):
        delta = self.compute_prediction_error(reward)
        if delta >= 0:
            self.weights = self.weights + self.learning_rate * delta * self.eligibility_trace
        if delta < 0:
            self.weights = self.weights + self.negative_learning_rate * delta * self.eligibility_trace

    def compute_prediction_error(self, reward):
        if reward == 1:
            delta = reward - self.previous_value
        else:
            delta = self.gamma * self.value - self.previous_value
        return delta

    def get_max_response(self):
        """Get the maximum place cell response.
        """
        return utils.gauss2d([0, 0], self.field_width ** 2, [0, 0])

    def choose_action(self):
        available_actions = self.get_available_actions()

        action_goodness = []
        for act in available_actions:
            next_x, next_y = self.env.compute_new_position(act)
            goal_cell_rate_diff = self.get_value(next_x, next_y)
            action_goodness.append(goal_cell_rate_diff)

        action_idx = utils.random_argmax(action_goodness)

        allocentric_action = available_actions[action_idx]
        # Back to egocentric reference frame
        egocentric_action = get_relative_angle(allocentric_action, self.env.curr_orientation)
        return egocentric_action, action_goodness[action_idx]

    def get_value(self, x, y):
        place_cell_responses = self.get_place_cell_response(x, y)
        goal_cell_rate = np.dot(self.weights, place_cell_responses)
        return goal_cell_rate

    def get_place_cell_response(self, x, y):
        """Compute the firing rate of a each place cell with Gaussian tuning curves, for the current position of the
        agent
        """
        if isinstance(self.env, PlusMaze):
            x, y = self.env.lookup_geodesic_coordinate([x, y])
            place_cell_responses = utils.gauss2d((x, y), var=self.field_width ** 2,
                                                 centre=self.geodesic_field_centres)
            place_cell_responses /= np.linalg.norm(place_cell_responses)
            return place_cell_responses

        place_cell_responses = utils.gauss2d((x, y), var=self.field_width ** 2,
                                             centre=self.field_centres) / self.max_response
        place_cell_responses /= np.linalg.norm(place_cell_responses)
        return place_cell_responses

    def get_available_actions(self):
        """Return those angles for which the absolute angle with the current direction is smaller than the max angle.
        """
        available_actions = [d for d in self.allocentric_directions
                             if abs(get_relative_angle(self.env.curr_orientation, d)) <= self.max_turning_angle]
        return available_actions

    def evaluate_value_function_everywhere(self):
        xs = np.linspace(self.env.minx, self.env.maxx, 50)
        ys = np.linspace(self.env.miny, self.env.maxy, 50)

        z = np.zeros((xs.shape[0], ys.shape[0]))

        for i in range(xs.shape[0]):
            for j in range(ys.shape[0]):
                z[i, j] = self.get_value(xs[i], ys[j])

        x, y = np.meshgrid(xs, ys)
        return x, y, z

    def plot_value_function(self):
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        x, y, z = self.evaluate_value_function_everywhere()
        ax.plot_surface(x, y, z, linewidth=0, antialiased=False, cmap=plt.cm.coolwarm)


class Hippocampus(object):
    """This class implements a simple model of hippocampus containing place cells and one goal cell that learn
    through Hebbian association.
    """

    def __init__(self, env=WaterMazeEnv(), learning_rate=.5, negative_learning_rate=.05, goal_cell_decay_factor=.99):
        self.env = env
        self.learning_rate = learning_rate
        self.negative_learning_rate = negative_learning_rate
        self.goal_cell_decay_factor = goal_cell_decay_factor
        self.max_turning_angle = 60

        # Create the place cells:
        self.field_width = .3
        self.field_centres = self.create_place_cells()
        if isinstance(self.env, PlusMaze):
            self.geodesic_field_centres = self.env.lookup_geodesic_coordinate(self.field_centres.T).T
        self.n_place_cells = self.field_centres.shape[1]
        self.max_response = self.get_max_response()
        self.previous_place_cell_responses = None
        self.place_cell_responses = None
        self.update_place_cell_response()

        # And the goal cell
        self.goal_cell_rate = 0
        self.weights = np.zeros(self.n_place_cells)
        if isinstance(self.env, PlusMaze):
            self.max_goal_response = .05
        else:
            self.max_goal_response = 15 #5
            self.cur_max = self.max_goal_response

        # Allocentric actions (0 is east)
        # Note that the only actions currently available to the agent will be -60 to 60 degrees away.
        self.allocentric_directions = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]

        self.remembered_goal_location = [0, 0]

    def choose_action(self):
        available_actions = self.get_available_actions()

        action_goodness = []
        for act in available_actions:
            next_x, next_y = self.env.compute_new_position(act)
            goal_cell_rate_diff = self.get_goal_cell_rate(next_x, next_y) - self.goal_cell_rate
            action_goodness.append(goal_cell_rate_diff)

        #if np.all(np.array(action_goodness) < 0):  # top reached but not goal
        #    self.weights *= 0

        action_idx = utils.random_argmax(action_goodness)

        allocentric_action = available_actions[action_idx]
        # Back to egocentric reference frame
        egocentric_action = get_relative_angle(allocentric_action, self.env.curr_orientation)
        return egocentric_action, action_goodness[action_idx]

    def get_available_actions(self):
        """Return those angles for which the absolute angle with the current direction is smaller than the max angle.
        """
        available_actions = [d for d in self.allocentric_directions
                             if abs(get_relative_angle(self.env.curr_orientation, d)) <= self.max_turning_angle]
        return available_actions

    def create_place_cells(self):
        """Create a grid of evenly spaced place field centres within the water maze.
        """
        field_centres_x, field_centres_y = np.meshgrid(
            np.linspace(self.env.minx, self.env.maxx, 12), np.linspace(self.env.miny, self.env.maxy, 12))
        field_centres = np.array([field_centres_x.flatten(), field_centres_y.flatten()])
        field_centres = field_centres[:, self.env.within_maze(field_centres)]
        return field_centres

    def on_platform(self):
        """Check if agent's current position is on the platform.

        :return (bool): True if agent on platform, false otherwise.
        """
        return (self.env.curr_x - self.env.platform_centre[0]) ** 2 + \
               (self.env.curr_y - self.env.platform_centre[1]) ** 2 < self.env.platform_radius ** 2

    def update_place_cell_response(self):
        """Compute the firing rate of a each place cell with Gaussian tuning curves, for the current position of the
        agent
        """
        self.previous_place_cell_responses = self.place_cell_responses
        if isinstance(self.env, PlusMaze):
            x, y = self.env.lookup_geodesic_coordinate([self.env.curr_x, self.env.curr_y])
            self.place_cell_responses = utils.gauss2d([x, y], var=self.field_width ** 2,
                                                      centre=self.geodesic_field_centres) / self.max_response
        else:
            self.place_cell_responses = utils.gauss2d([self.env.curr_x, self.env.curr_y], var=self.field_width ** 2,
                                                      centre=self.field_centres) / self.max_response

    def get_place_cell_response(self, x, y):
        """Compute the firing rate of a each place cell with Gaussian tuning curves, for the current position of the
        agent
        """
        if isinstance(self.env, PlusMaze):
            x, y = self.env.lookup_geodesic_coordinate([x, y])
            place_cell_responses = utils.gauss2d((x, y), var=self.field_width ** 2,
                                                 centre=self.geodesic_field_centres) / self.max_response
            return place_cell_responses

        place_cell_responses = utils.gauss2d((x, y), var=self.field_width ** 2,
                                             centre=self.field_centres) / self.max_response
        return place_cell_responses

    def get_max_response(self):
        """Get the maximum place cell response.
        """
        return utils.gauss2d([0, 0], self.field_width ** 2, [0, 0])

    def update_weights(self):
        """Update place to goal cell weights using a Hebbian learning rule.
        """
        delta_weights = self.learning_rate * self.goal_cell_rate * self.place_cell_responses
        self.weights += delta_weights
        self.normalise_weights()

        xs, ys, goal_surface = self.evaluate_value_function_everywhere()
        j, i = np.unravel_index(goal_surface.argmax(), xs.shape)
        self.remembered_goal_location = [xs[i, j], ys[i, j]]

    def normalise_weights(self):
        goal_cell_response = self.get_goal_cell_rate(self.env.curr_x, self.env.curr_y)
        if goal_cell_response > self.max_goal_response:
            self.weights = self.weights / goal_cell_response * self.max_goal_response

    def compute_goal_cell_rate(self):
        self.goal_cell_rate = np.dot(self.weights, self.place_cell_responses)

    def get_goal_cell_rate(self, x, y):
        place_cell_responses = self.get_place_cell_response(x, y)
        goal_cell_rate = np.dot(self.weights, place_cell_responses)
        return goal_cell_rate

    def compute_goal_surface(self):
        pass

    def evaluate_value_function_everywhere(self):
        xs = np.linspace(self.env.minx, self.env.maxx, 50)
        ys = np.linspace(self.env.miny, self.env.maxy, 50)

        z = np.zeros((xs.shape[0], ys.shape[0]))

        for i in range(xs.shape[0]):
            for j in range(ys.shape[0]):
                z[i, j] = self.get_goal_cell_rate(xs[i], ys[j])

        x, y = np.meshgrid(xs, ys)
        return x, y, z

    def plot_value_function(self):
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        x, y, z = self.evaluate_value_function_everywhere()
        ax.plot_surface(x, y, z, linewidth=0, antialiased=False, cmap=plt.cm.coolwarm)

    def update(self):
        self.update_place_cell_response()
        self.compute_goal_cell_rate()
        if distance.euclidean([self.env.curr_x, self.env.curr_y],
                                                          self.remembered_goal_location) < self.env.platform_radius:  #(self.max_goal_response * .999):
            self.weights -= self.negative_learning_rate * self.goal_cell_rate * self.place_cell_responses
            self.weights = np.maximum(self.weights, np.zeros(len(self.weights)))
        #self.weights *= self.goal_cell_decay_factor

    def goal_reached(self):
        """Set the goal cell rate to maximum, and learn weights with place cells to remember
        """
        self.goal_cell_rate = self.max_goal_response
        self.update_weights()


class CognitiveHippocampus(object):
    def __init__(self, env=DeterministicTask(), learning_rate=.5, gamma=.9):
        self.learning_rate = learning_rate
        self.env = env
        self.gamma = gamma
        self.previous_place_cell_responses = None
        self.place_cell_responses = None
        self.place_cell_matrix = self.get_place_cell_mat()
        self.weights = np.zeros(self.env.nr_states)
        self.goal_cell_rate = 0

        self.max_goal_response = 1

    def choose_action(self):
        available_actions = self.env.get_possible_actions(self.env.curr_state)
        action_goodness = []
        for act in available_actions:

            possible_next_states, probabilities = self.env.get_possible_next_states(self.env.curr_state, act)
            next_state = possible_next_states[np.argmax(probabilities)]

            goal_cell_rate_diff = self.get_goal_cell_rate(next_state) - self.goal_cell_rate
            action_goodness.append(goal_cell_rate_diff)

        probs = utils.softmax(action_goodness, beta=4)
        action_idx = np.random.choice(np.arange(len(available_actions)), p=probs)
        #action_idx = utils.random_argmax(action_goodness)
        return available_actions[action_idx], action_goodness[action_idx]

    def update(self):
        self.update_place_cell_response()
        self.compute_goal_cell_rate()
        # self.weights *= self.goal_cell_decay_factor

    def compute_goal_cell_rate(self):
        self.goal_cell_rate = np.dot(self.weights, self.place_cell_responses)

    def get_place_cell_mat(self):
        """Compute the place cell matrix (population responses in the rows for each state).
        :return:
        """
        rw_transition_matrix = self.env.transition_probabilities.mean(axis=1)
        successor_rep = np.linalg.inv(np.eye(self.env.nr_states) - self.gamma * rw_transition_matrix)
        return make_symmetric(successor_rep)

    def update_place_cell_response(self):
        self.previous_place_cell_responses = self.place_cell_responses
        self.place_cell_responses = self.place_cell_matrix[self.env.curr_state]

    def get_goal_cell_rate(self, state_idx):
        place_cell_responses = self.place_cell_matrix[state_idx]
        goal_cell_rate = np.dot(self.weights, place_cell_responses)
        return goal_cell_rate

    def get_goal_surface(self):
        goal_surface = []
        for state in range(self.env.nr_states):
            goal_cell_rate = self.get_goal_cell_rate(state)
            goal_surface.append(goal_cell_rate)
        return goal_surface

    def update_weights(self):
        """Update place to goal cell weights using a Hebbian learning rule.
        """
        delta_weights = self.learning_rate * self.goal_cell_rate * self.place_cell_responses
        self.weights = delta_weights
        self.normalise_weights()

    def normalise_weights(self):
        goal_cell_response = self.get_goal_cell_rate(self.env.curr_state)
        if goal_cell_response > self.max_goal_response:
            self.weights = self.weights / goal_cell_response * self.max_goal_response


if __name__ == '__main__':
    h = Hippocampus(env=PlusMaze())
    h.update_place_cell_response()

    h.goal_cell_rate = 1
    h.update_weights()
    h.plot_value_function()

    h.env.curr_x, h.env.curr_y = [.5, .5]
    h.update_place_cell_response()
    h.compute_goal_cell_rate()

    h.choose_action()

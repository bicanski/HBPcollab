import numpy as np
from environments import WaterMazeEnv
import utils
from utils import get_relative_angle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class TDHippocampus(object):
    """This class implements a model hippocampus that uses TD learning to learn a value function using geodesic place
    cell features as a basis set, following Gustafson & Daw (2011).
    """
    def __init__(self, env=WaterMazeEnv(), learning_rate=.02, negative_learning_rate=.02, gamma=.98, lamb=.67):
        self.env = env
        self.learning_rate = learning_rate
        self.negative_learning_rate = negative_learning_rate
        self.gamma = gamma
        self.lamb = lamb
        self.max_turning_angle = 60

        # Create the place cells:
        self.field_width = .3
        self.field_centres = self.create_place_cells()
        self.n_place_cells = self.field_centres.shape[1]
        self.max_response = self.get_max_response()
        self.previous_place_cell_responses = None
        self.place_cell_responses = None
        self.update_place_cell_response()

        # And the goal cell
        self.value = 0
        self.previous_value = 0

        self.weights = np.zeros(self.place_cell_responses.shape)
        #self.weights = np.random.rand(self.n_place_cells) * .0004
        self.eligibility_trace = np.zeros(self.weights.shape)

        # Allocentric actions (0 is east)
        # Note that the only actions currently available to the agent will be -60 to 60 degrees away.
        self.allocentric_directions = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
        self.allocentric_directions = self.env.all_orientations
        self.remembered_goal_location = [0, 0]

    def create_place_cells(self):
        """Create a grid of evenly spaced place field centres within the water maze.
        """
        field_centres_x, field_centres_y = np.meshgrid(
            np.linspace(self.env.minx, self.env.maxx, 40), np.linspace(self.env.miny, self.env.maxy, 40))
        field_centres = np.array([field_centres_x.flatten(), field_centres_y.flatten()])
        field_centres = field_centres[:, self.env.within_maze(field_centres)]
        return field_centres

    def update_place_cell_response(self):
        """Compute the firing rate of a each place cell with Gaussian tuning curves, for the current position of the
        agent
        """
        self.previous_place_cell_responses = self.place_cell_responses
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
            goal_cell_rate_diff = self.get_value(next_x, next_y) #- self.get_value(self.env.curr_x, self.env.curr_y)
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

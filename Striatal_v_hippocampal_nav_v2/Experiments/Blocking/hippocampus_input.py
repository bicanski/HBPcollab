import numpy as np
from Experiments.Blocking.environment import Environment
from itertools import product
import os

from utils import constrained_poisson_sample, get_relative_angle, random_argmax
from definitions import ROOT_FOLDER


bvc_maps_folder = os.path.join(ROOT_FOLDER, 'Data/BVC_ratemaps/')
ovc_maps_folder = os.path.join(ROOT_FOLDER, 'Data/OVC_ratemaps/')


class PlaceCell(object):
    def __init__(self, env=Environment()):
        self.env = env
        self.threshold = 2  # 12
        self.xs, self.ys = np.meshgrid(self.env.xs, self.env.ys)
        self.cues_present = self.env.cues_present
        self.n_bvc_inputs = constrained_poisson_sample(6, 4, 20)
        self.n_ovc_inputs = constrained_poisson_sample(2, 0, 8)
        self.bvc_idx = np.arange(self.n_bvc_inputs)
        self.ovc_idx = np.arange(self.n_bvc_inputs, self.n_bvc_inputs + self.n_ovc_inputs)

        self.bvc_maps = self.load_BVCs()
        self.ovc_maps = self.load_OVCs()
        self.input_cells = np.array(self.bvc_maps + self.ovc_maps)

    def change_cues(self):
        self.cues_present = self.env.cues_present
        self.ovc_maps = self.load_OVCs()
        self.input_cells = np.array(self.bvc_maps + self.ovc_maps)

    def get_input_current(self, x, y):
        # Use KD tree or something to get closest location in grid
        # Then look up that location in every map. Probably most useful if all maps in one big array (just slice)

        ix = int(x/10)
        iy = int(y/10)
        if ix >= 100:
            ix = 100
        if iy >= 100:
            iy = 100

        input_vec = self.input_cells[:, ix, iy]
        input_current = max(0, np.sum(input_vec) - self.threshold)
        return input_current

    def load_BVCs(self):
        pref_distances = np.random.choice([81.0, 169.0, 265.0, 369.0, 482.5, 606.5], self.n_bvc_inputs)
        pref_orientations = np.random.choice(np.linspace(0, 360, 61)[:-1], self.n_bvc_inputs)
        bvc_maps = []
        for d, o in zip(pref_distances, pref_orientations):
            fname = 'bvcmap_{}_distance{}_orientation{}.npy'.format('Square1x1', int(d), int(np.round(o)))
            rate_map = np.load(os.path.join(bvc_maps_folder, fname))
            bvc_maps.append(rate_map)
        return bvc_maps

    def load_OVCs(self):
        pref_distances = np.random.choice([169.0, 265.0, 369.0, 482.5], self.n_ovc_inputs)
        pref_orientations = np.random.choice(np.linspace(0, 360, 61)[:-1], self.n_ovc_inputs)
        ovc_maps = []
        for d, o in zip(pref_distances, pref_orientations):
            fname = 'ovcmap_{}_distance{}_orientation{}.npy'.format(self.cues_present, int(d), int(np.round(o)))
            rate_map = np.load(os.path.join(ovc_maps_folder, fname))
            ovc_maps.append(rate_map)
        return ovc_maps

    @staticmethod
    def load_BVC_maps(distances, orientations):
        all_bvc_maps = {d: {} for d in distances}
        for d, o in product(distances, orientations):
            fname = 'bvcmap_{}_distance{}_orientation{}.npy'.format('Square1x1', int(d), int(o))
            bvc_map = np.load(os.path.join(bvc_maps_folder, fname))
            all_bvc_maps[d][o] = bvc_map
        return all_bvc_maps

    def get_input_ratemap(self):
        ratemap = np.empty((self.env.xs.shape[0], self.env.ys.shape[0]))
        for ix, x in enumerate(tqdm(self.env.xs)):
            for iy, y in enumerate(self.env.ys):
                activation = self.get_input_current(x, y)
                ratemap[ix, iy] = activation
        return ratemap


class Hippocampus(object):
    def __init__(self, env=Environment(), learning_rate=.005, gamma=.98, lamb=.67):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lamb = lamb
        self.max_turning_angle = 60

        # Create the place cells:
        self.n_place_cells = 40
        self.place_cells = [PlaceCell() for _ in range(self.n_place_cells)]
        self.previous_place_cell_responses = None
        self.place_cell_responses = None
        self.update_place_cell_response()

        # And the goal cell
        self.value = 0
        self.previous_value = 0

        self.weights = np.zeros(self.n_place_cells)
        # self.weights = np.random.rand(self.n_place_cells) * .04
        self.eligibility_trace = np.zeros(self.weights.shape)

        # Allocentric actions (0 is east)
        # Note that the only actions currently available to the agent will be -60 to 60 degrees away.
        self.allocentric_directions = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]

    def update_place_cell_response(self):
        """Compute the firing rate of a each place cell with Gaussian tuning curves, for the current position of the
        agent
        """
        self.previous_place_cell_responses = self.place_cell_responses
        self.place_cell_responses = [pc.get_input_current(self.env.curr_x, self.env.curr_y) for pc in self.place_cells]
        #self.place_cell_responses = softmax(self.place_cell_responses, beta=2)
        self.place_cell_responses /= np.linalg.norm(self.place_cell_responses)

    def get_place_cell_response(self, x, y):
        """Compute the firing rate of a each place cell with Gaussian tuning curves, for the current position of the
        agent
        """

        place_cell_responses = [pc.get_input_current(x, y) for pc in self.place_cells]
        #place_cell_responses = softmax(place_cell_responses, beta=2)
        if np.linalg.norm(place_cell_responses) != 0:
            place_cell_responses /= np.linalg.norm(place_cell_responses)
        return place_cell_responses

    def get_ratemap(self):
        rate_maps = np.empty((self.n_place_cells, len(self.env.xs), len(self.env.ys)))
        for ix, x in enumerate(self.env.xs):
            for iy, y in enumerate(self.env.ys):
                place_cell_responses = self.get_place_cell_response(x, y)
                rate_maps[:, ix, iy] = place_cell_responses
        return rate_maps

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

    def compute_prediction_error(self, reward):
        if reward == 1:
            delta = reward - self.previous_value
        else:
            delta = self.gamma * self.value - self.previous_value
        return delta

    def choose_action(self):
        available_actions = self.get_available_actions()

        action_goodness = []
        for act in available_actions:
            next_x, next_y = self.env.compute_new_position(act)
            val = self.get_value(next_x, next_y)
            action_goodness.append(val)

        # hack: for every nan, give negative value
        action_goodness = np.array(action_goodness)
        action_goodness[np.isnan(action_goodness)] = -100

        action_idx = random_argmax(action_goodness)

        allocentric_action = available_actions[action_idx]
        # Back to egocentric reference frame
        egocentric_action = get_relative_angle(allocentric_action, self.env.curr_orientation)
        return egocentric_action, action_goodness[action_idx]

    def get_value(self, x, y):
        place_cell_responses = self.get_place_cell_response(x, y)
        goal_cell_rate = np.dot(self.weights, place_cell_responses)
        return goal_cell_rate

    def get_available_actions(self):
        """Return those angles for which the absolute angle with the current direction is smaller than the max angle.
        """
        available_actions = [d for d in self.allocentric_directions
                             if abs(get_relative_angle(self.env.curr_orientation, d)) <= self.max_turning_angle]
        return available_actions

    def evaluate_value_function_everywhere(self):
        xs = np.linspace(0, 1000, 50)
        ys = np.linspace(0, 1000, 50)

        z = np.zeros((xs.shape[0], ys.shape[0]))

        for i in range(xs.shape[0]):
            for j in range(ys.shape[0]):
                z[i, j] = self.get_value(xs[i], ys[j])

        x, y = np.meshgrid(xs, ys)
        return x, y, z


if __name__ == '__main__':
    from tqdm import tqdm
    import matplotlib.pyplot as plt

    n_place_cells = 30
    pcs = [PlaceCell() for i in range(n_place_cells)]

    xs = np.arange(5, 1010, 10)
    ys = np.arange(5, 1010, 10)

    rate_map = np.empty((len(pcs), len(xs), len(ys)))
    for ix, x in enumerate(tqdm(xs)):
        for iy, y in enumerate(ys):
            activation = [pc.get_input_current(x, y) for pc in pcs]
            #activation = softmax(activation, beta=2)
            rate_map[:, ix, iy] = activation

    plt.imshow(rate_map[0])
    plt.show()

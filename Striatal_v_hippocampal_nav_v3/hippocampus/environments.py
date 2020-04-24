import numpy as np
import networkx as nx
import pandas as pd
import random
import os
from definitions import ROOT_FOLDER
from hippocampus import utils
from hippocampus.dynamic_programming_utils import generate_random_policy
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon, Rectangle
from matplotlib import collections as mc
import seaborn as sns
from scipy.spatial.distance import euclidean
from itertools import product
from tqdm import tqdm
import pyximport
from copy import deepcopy
from multiprocessing import Pool
pyximport.install(setup_args={'include_dirs': np.get_include()})

from hippocampus.fastBVC import Boundary, BVC
#from hippocampus.slowBVC import BVC


class Environment(object):
    """Parent class for RL environments holding some general methods.
    """
    def __init__(self):
        self.nr_states = None
        self.nr_actions = None
        self.actions = None
        self.adjacency_graph = None
        self.goal_state = None
        self.reward_func = None
        self.graph = None
        self.n_features = None
        self.rf = None
        self.transition_probabilities = None
        self.terminal_state = None
        self.state_indices = None
        self.current_state = None
        self.landmark_location = None
        self.grid = None
        self.ego_angles = None
        self.allo_angles = None
        self.action_directions = None

    def act(self, action):
        pass

    def get_goal_state(self):
        pass

    def get_current_state(self):
        return self.current_state

    def reset(self):
        pass

    def define_adjacency_graph(self):
        pass

    def _fill_adjacency_matrix(self):
        pass

    def get_adjacency_matrix(self):
        if self.adjacency_graph is None:
            self._fill_adjacency_matrix()
        return self.adjacency_graph

    def create_graph(self):
        """Create networkx graph from adjacency matrix.
        """
        self.graph = nx.from_numpy_array(self.get_adjacency_matrix())

    def show_graph(self, map_variable=None, layout=None, node_size=1500, **kwargs):
        """Plot graph showing possible state transitions.

        :param node_size:
        :param map_variable: Continuous variable that can be mapped on the node colours.
        :param layout:
        :param kwargs: Any other drawing parameters accepted. See nx.draw docs.
        :return:
        """
        if layout is None:
            layout = nx.spring_layout(self.graph)
        if map_variable is not None:
            categories = pd.Categorical(map_variable)
            node_color = categories
        else:
            node_color = 'b'
        nx.draw(self.graph, with_labels=True, pos=layout, node_color=node_color, node_size=node_size, **kwargs)

    def set_reward_location(self, state_idx, action_idx):
        self.goal_state = state_idx
        action_destination = self.transition_probabilities[state_idx, action_idx]
        self.reward_func = np.zeros([self.nr_states, self.nr_actions, self.nr_states])
        self.reward_func[state_idx, action_idx] = action_destination

    def is_terminal(self, state_idx):
        if not self.get_possible_actions(state_idx):
            return True
        else:
            return False

    def get_destination_state(self, current_state, current_action):
        transition_probabilities = self.transition_probabilities[current_state, current_action]
        return np.flatnonzero(transition_probabilities)

    def get_degree_mat(self):
        degree_mat = np.eye(self.nr_states)
        for state, degree in self.graph.degree:
            degree_mat[state, state] = degree
        return degree_mat

    def get_laplacian(self):
        return self.get_degree_mat() - self.adjacency_graph

    def get_normalised_laplacian(self):
        """Return the normalised laplacian.
        """
        D = self.get_degree_mat()
        L = self.get_laplacian()  # TODO: check diff with non normalised laplacian. check adverserial examples
        exp_D = utils.exponentiate(D, -.5)
        return exp_D.dot(L).dot(exp_D)

    def compute_laplacian(self, normalization_method=None):
        """Compute the Laplacian.

        :param normalization_method: Choose None for unnormalized, 'rw' for RW normalized or 'sym' for symmetric.
        :return:
        """
        if normalization_method not in [None, 'rw', 'sym']:
            raise ValueError('Not a valid normalisation method. See help(compute_laplacian) for more info.')

        D = self.get_degree_mat()
        L = D - self.adjacency_graph

        if normalization_method is None:
            return L
        elif normalization_method == 'sym':
            exp_D = utils.exponentiate(D, -.5)
            return exp_D.dot(L).dot(exp_D)
        elif normalization_method == 'rw':
            exp_D = utils.exponentiate(D, -1)
            return exp_D.dot(L)

    def get_possible_actions(self, state_idx):
        pass

    def get_adjacent_states(self, state_idx):
        pass

    def compute_feature_response(self):
        pass

    def get_transition_matrix(self, policy):
        transition_matrix = np.zeros([self.nr_states, self.nr_states])
        for state in self.state_indices:
            if self.is_terminal(state):
                continue
            actions = self.get_possible_actions(state)
            for a, action in enumerate(actions):
                transition_matrix[state] += self.transition_probabilities[state, a] * policy[state][a]
        return transition_matrix

    def get_successor_representation(self, policy, gamma=.95):
        transition_matrix = self.get_transition_matrix(policy)
        m = np.linalg.inv(np.eye(self.nr_states) - gamma * transition_matrix)
        return m

    def get_next_state(self, state_idx, a):
        pass

    def get_state_location(self, idx):
        pass


class SimpleMDP(Environment):
    """Very simple MDP with states on a linear track. Agent gets reward of 1 if it reaches last state.
    """

    def __init__(self, nr_states=3, reward_probability=1.):
        Environment.__init__(self)
        self.reward_probability = reward_probability
        self.nr_states = nr_states
        self.state_indices = np.arange(self.nr_states)
        self.nr_actions = 2
        self.actions = [0, 1]
        self.action_consequences = {0: -1, 1: +1}
        self.terminal_states = [self.nr_states - 1]

        self.transition_probabilities = self.define_transition_probabilities()
        self.reward_func = np.zeros((self.nr_states, self.nr_actions))
        self.reward_func[self.nr_states - 2, 1] = 1
        self.current_state = 0

    def reset(self):
        self.current_state = 0

    def define_transition_probabilities(self):
        transition_probabilities = np.zeros([self.nr_states, self.nr_actions, self.nr_states])
        for predecessor in self.state_indices:
            if self.is_terminal(predecessor):
                transition_probabilities[predecessor, :, :] = 0
                continue
            for action_key, consequence in self.action_consequences.items():
                successor = int(predecessor + consequence)
                if successor not in self.state_indices:
                    transition_probabilities[predecessor, action_key, predecessor] = 1  # stay in current state
                else:
                    transition_probabilities[predecessor, action_key, successor] = 1
        return transition_probabilities

    def get_possible_actions(self, state_idx):
        if state_idx in self.terminal_states:
            return []
        else:
            return list(self.action_consequences)

    def define_adjacency_graph(self):
        transitions_under_random_policy = self.transition_probabilities.sum(axis=1)
        adjacency_graph = transitions_under_random_policy != 0
        return adjacency_graph.astype('int')

    def get_transition_matrix(self, policy):
        transition_matrix = np.zeros([self.nr_states, self.nr_states])
        for state in self.state_indices:
            if self.is_terminal(state):
                continue
            for action in range(self.nr_actions):
                transition_matrix[state] += self.transition_probabilities[state, action] * policy[state][action]
        return transition_matrix

    def get_successor_representation(self, policy, gamma=.95):
        transition_matrix = self.get_transition_matrix(policy)
        m = np.linalg.inv(np.eye(self.nr_states) - gamma * transition_matrix)
        return m

    def get_next_state(self, current_state, action):
        next_state = np.flatnonzero(self.transition_probabilities[current_state, action])[0]
        return next_state

    def get_reward(self, current_state, action):
        if np.random.rand() <= self.reward_probability:
            reward = self.reward_func[current_state, action]
        else:
            reward = 0.
        return reward

    def get_next_state_and_reward(self, current_state, action):
        # If current state is terminal absorbing state:
        if self.is_terminal(current_state):
            return current_state, 0

        next_state = self.get_next_state(current_state, action)
        reward = self.get_reward(current_state, action)
        return next_state, reward

    def act(self, action):
        next_state, reward = self.get_next_state_and_reward(self.current_state, action)
        self.current_state = next_state
        return next_state, reward

    def get_current_state(self):
        """Return current state idx given current position.
        """
        return self.current_state

    def _fill_adjacency_matrix(self):
        self.adjacency_graph = np.zeros((self.nr_states, self.nr_states), dtype=np.int)
        for idx in self.state_indices:
            if (idx + 1) in self.state_indices:
                self.adjacency_graph[idx, idx + 1] = 1

    def get_adjacency_matrix(self):
        if self.adjacency_graph is None:
            self._fill_adjacency_matrix()
        return self.adjacency_graph


class HexGrid(object):
    def __init__(self, radius):
        self.deltas = [[1, 0, -1], [0, 1, -1], [-1, 1, 0], [-1, 0, 1], [0, -1, 1], [1, -1, 0]]
        self.radius = radius
        self.cube_coords = {0: (0, 0, 0)}
        self.edge_states = []
        state = 1
        for r in range(radius):
            a = 0
            b = -r
            c = +r
            for j in range(6):
                num_of_hexes_in_edge = r
                for i in range(num_of_hexes_in_edge):
                    a = a + self.deltas[j][0]
                    b = b + self.deltas[j][1]
                    c = c + self.deltas[j][2]
                    self.cube_coords[state] = (a, b, c)
                    if r == radius - 1:
                        self.edge_states.append(state)
                    state += 1

        self.cart_coords = {state: self.to_cartesian(coord) for state, coord in self.cube_coords.items()}
        self.size = len(self.cube_coords)

    def get_adjacency(self):
        adjacency_matrix = np.zeros((len(self.cube_coords), len(self.cube_coords)))
        for state, coord in self.cube_coords.items():
            for d in self.deltas:
                a = coord[0] + d[0]
                b = coord[1] + d[1]
                c = coord[2] + d[2]
                neighbour = self.get_state_id((a, b, c))
                if neighbour is not None:
                    adjacency_matrix[state, neighbour] = 1
        return adjacency_matrix

    def get_sas_transition_mat(self):
        """Fill and return the state by action by state transition matrix.

        :return:
        """
        sas_matrix = np.zeros((len(self.cube_coords), len(self.deltas), len(self.cube_coords)))
        for state, coord in self.cube_coords.items():
            for i, d in enumerate(self.deltas):
                a = coord[0] + d[0]
                b = coord[1] + d[1]
                c = coord[2] + d[2]
                neighbour = self.get_state_id((a, b, c))
                if neighbour is not None:
                    sas_matrix[state, i, neighbour] = 1.
                else:  # if a wall state is the neighbour
                    sas_matrix[state, i, state] = 1.
        return sas_matrix

    def get_state_id(self, cube_coordinate):
        for state, loc in self.cube_coords.items():
            if loc == cube_coordinate:
                return state
        return None

    def is_state_location(self, coordinate):
        """Return true if cube coordinate exists.

        :param coordinate: Tuple cube coordinate
        :return:
        """
        for state, loc in self.cube_coords.items():
            if loc == coordinate:
                return True
        return False

    @staticmethod
    def to_cartesian(coordinate):
        xcoord = coordinate[0]
        ycoord = 2. * np.sin(np.radians(60)) * (coordinate[1] - coordinate[2]) / 3.
        return xcoord, ycoord

    def plot_grid(self):
        fig, ax = plt.subplots(1)
        ax.set_aspect('equal')
        for x, y in self.cart_coords.values():
            hex_patch = RegularPolygon((x, y), numVertices=6, radius=2. / 3.,
                                       orientation=np.radians(30), alpha=0.2, edgecolor='k')
            ax.add_patch(hex_patch)

        lower_bound = min(min(self.cart_coords.values()))
        upper_bound = max(max(self.cart_coords.values()))
        plt.xlim([lower_bound - 2, upper_bound + 2])
        plt.ylim([lower_bound - 2, upper_bound + 2])
        return fig, ax

    def distance(self, state_a, state_b):
        return euclidean(self.cart_coords[state_a], self.cart_coords[state_b])


class HexWaterMaze(Environment):
    def __init__(self, radius):
        super().__init__()
        self.grid = HexGrid(radius)
        self.adjacency_graph = self.grid.get_adjacency()
        self.transition_probabilities = self.grid.get_sas_transition_mat()
        self.action_labels = ['N', 'NE', 'SE', 'S', 'SW', 'NW']
        self.actions = self.grid.deltas
        self.action_directions = []
        for a in self.actions:
            mv = self.grid.to_cartesian(a)
            self.action_directions.append(np.arctan2(mv[1], mv[0]))
        self.nr_actions = len(self.actions)
        self.nr_states = self.grid.size
        self.state_indices = list(range(self.nr_states))
        states_close_to_centre = [i for i in self.state_indices if
                                  euclidean(self.grid.cart_coords[i], self.grid.cart_coords[0]) < radius / 3]
        self.platform_state = np.random.choice([i for i in self.state_indices if not i in states_close_to_centre])
        self.previous_platform_state = None
        self.reward_func = np.zeros((self.nr_states, self.nr_actions, self.nr_states))
        self.set_reward_func()
        self.landmark_location = self.set_landmark()
        self.starting_state = 0
        self.allo_angles = np.array([30, 90, 150, 210, 270, 330])
        self.ego_angles = np.array([0, 60, 120, -180, -120, -60])  # q values correspond to these

    def set_platform_state(self, state_idx):
        self.previous_platform_state = self.platform_state
        self.platform_state = state_idx
        self.landmark_location = self.set_landmark()
        self.set_reward_func()

    def set_reward_func(self):
        for state in self.state_indices:
            for action in range(self.nr_actions):
                next_state, reward = self.get_next_state_and_reward(state, action)
                self.reward_func[state, action, next_state] = reward

    def set_landmark(self):
        platform_loc = self.grid.cube_coords[self.platform_state]
        landmark_loc = (platform_loc[0], platform_loc[1] + 1, platform_loc[2])
        return self.grid.to_cartesian(landmark_loc)

    def get_next_state(self, current_state, action):
        next_state = np.flatnonzero(self.transition_probabilities[current_state, action])[0]
        return next_state

    def get_reward(self, next_state):
        if next_state == self.platform_state:
            return 1.
        else:
            return 0.

    def get_next_state_and_reward(self, current_state, action):
        # If current state is terminal absorbing state:
        if self.is_terminal(current_state):
            return current_state, 0

        next_state = self.get_next_state(current_state, action)
        reward = self.get_reward(next_state)
        return next_state, reward

    def act(self, action):
        next_state, reward = self.get_next_state_and_reward(self.current_state, action)
        self.current_state = next_state
        return next_state, reward

    def reset(self, random_loc=True):
        if random_loc:
            dist_from_platform = np.array([self.grid.distance(self.platform_state, s) for s in self.grid.edge_states])
            eligible_start_states = np.array(self.grid.edge_states)[dist_from_platform > self.grid.radius * 1.5]
            self.starting_state = np.random.choice(eligible_start_states)
        self.current_state = self.starting_state

    def get_state_location(self, state, cube_system=False):
        if cube_system:
            return self.grid.cube_coords[state]
        else:
            return self.grid.cart_coords[state]

    def get_transition_matrix(self, policy):
        transition_matrix = np.zeros([self.nr_states, self.nr_states])
        for state in self.state_indices:
            if self.is_terminal(state):
                continue
            for action in range(self.nr_actions):
                transition_matrix[state] += self.transition_probabilities[state, action] * policy[state][action]
        return transition_matrix

    def get_successor_representation(self, policy, gamma=.95):
        """Compute the Successor Representation through inversion of the transition matrix.

        :param (list) policy: Nested list containing the action probabilities for each state.
        :param (float) gamma: Discount parameter
        :return:
        """
        transition_matrix = self.get_transition_matrix(policy)
        m = np.linalg.inv(np.eye(self.nr_states) - gamma * transition_matrix)
        return m

    def get_possible_actions(self, state_idx):
        if self.is_terminal(state_idx):
            return []
        else:
            return list(range(self.nr_actions))

    def is_terminal(self, state_idx):
        if state_idx == self.platform_state:
            return True
        else:
            return False

    def plot_grid(self, c_mappable=None, ax=None, show_state_idx=True, alpha=1., c_map=None):
        """

        :param show_state_idx:
        :param (np.array) c_mappable:
        :return:
        """
        # TODO: move to plotting module, make class for plotting hex grids.
        if ax is None:
            fig, ax = plt.subplots()
        ax.set_aspect('equal')

        if c_mappable is not None:
            if c_map is None:
                cmap = plt.get_cmap('Greys_r')
            else:
                cmap = plt.get_cmap(c_map)
            c_mappable = c_mappable / c_mappable.max()

        for i, (x, y) in enumerate(self.grid.cart_coords.values()):
            if c_mappable is not None:
                colour = cmap(c_mappable[i])
            else:
                colour = 'gray'
            hex_patch = RegularPolygon((x, y), numVertices=6, radius=2. / 3., facecolor=colour,
                                       orientation=np.radians(30), alpha=alpha, edgecolor='k')
            ax.add_patch(hex_patch)
            if show_state_idx:
                ax.text(x, y, str(i), ha='center', va='center', size=10)

        lower_bound = min(min(self.grid.cart_coords.values()))
        upper_bound = max(max(self.grid.cart_coords.values()))
        plt.xlim([lower_bound - 2, upper_bound + 2])
        plt.ylim([lower_bound - 2, upper_bound + 2])
        return ax

    def plot_occupancy_on_grid(self, trial_results, **kwargs):
        color_pal = sns.color_palette()
        state_occupancy = trial_results['state'].astype(int).value_counts()
        occupancy = np.zeros(self.nr_states)
        for s, count in state_occupancy.iteritems():
            occupancy[s] = count
        ax = self.plot_grid(occupancy, **kwargs)

        platform = trial_results['platform'].iloc[0]
        previous_platform = trial_results['previous platform'].iloc[0]

        hex_patch = RegularPolygon(self.grid.cart_coords[platform], numVertices=6, radius=2./2.5,
                                   facecolor=color_pal[8], orientation=np.radians(30), alpha=1., edgecolor='k')
        ax.add_patch(hex_patch)
        hex_patch = RegularPolygon(self.grid.cart_coords[previous_platform], numVertices=6, radius=2./2.5,
                                   facecolor=color_pal[9], orientation=np.radians(30), alpha=1., edgecolor='k')
        ax.add_patch(hex_patch)

        # add legend
        hex_patch = RegularPolygon((6, 10), numVertices=6, radius=2. / 3.,
                                   facecolor=color_pal[8],
                                   orientation=np.radians(30), alpha=1., edgecolor='k')
        ax.add_patch(hex_patch)
        hex_patch = RegularPolygon((6, 8.5), numVertices=6, radius=2. / 3.,
                                   facecolor=color_pal[9],
                                   orientation=np.radians(30), alpha=1., edgecolor='k')
        ax.add_patch(hex_patch)

        ax.text(x=7, y=10, s='Platform', va='center')
        ax.text(x=7, y=8.5, s='Previous platform', va='center')

        start = trial_results['state'].iloc[0]
        hex_patch = RegularPolygon(self.grid.cart_coords[start], numVertices=6, radius=2./2.5,
                                   facecolor=color_pal[2],
                                   orientation=np.radians(30), alpha=1., edgecolor='k')
        ax.add_patch(hex_patch)
        ax.text(self.grid.cart_coords[start][0], self.grid.cart_coords[start][1], 'S',
                ha='center', va='center', size=10)

        ax.axis('off')
        return ax

    def get_goal_state(self):
        return self.platform_state


class PlusMaze(Environment):
    """Packard & McGaugh experiment: start in same start arm for 3 trials, then probe trial in opposite.
    """
    def __init__(self):
        super().__init__()
        self.actions = ['N', 'E', 'S', 'W']
        self.allo_angles = np.array([90, 0, 270, 180])
        self.ego_angles = np.array([0, 90, -180, -90])
        self.ego_actions = ['F', 'L', 'B', 'R']
        self.action_directions = np.radians([90, 0, 270, 180])
        self.nr_actions = len(self.actions)

        self.states_actions_outcomes = {
            0: {},
            1: {'N': 2, 'E': 1, 'S': 1, 'W': 1},
            2: {'N': 4, 'E': 5, 'S': 1, 'W': 3},
            3: {'N': 0, 'E': 0, 'S': 0, 'W': 0},
            4: {'N': 4, 'E': 4, 'S': 2, 'W': 4},
            5: {'N': 0, 'E': 0, 'S': 0, 'W': 0},
        }

        self.state_locs = {
            0: (np.nan, np.nan),
            1: (1, 0),
            2: (1, 1),
            3: (0, 1),
            4: (1, 2),
            5: (2, 1),
        }

        self.all_locs = np.array([val for key, val in self.state_locs.items() if key != 0])

        max_dist = 0
        for i, j in product(self.all_locs, self.all_locs):
            d = euclidean(i, j)
            if d > max_dist:
                max_dist = d
        self.max_dist = max_dist
        self.center = self.state_locs[2]
        self.nr_states = len(self.states_actions_outcomes)
        self.state_indices = np.arange(self.nr_states)
        self.landmark_location = self.get_state_location(4)

        self._fill_adjacency_matrix()
        self.start_state = 1
        self.current_state = self.start_state
        self.rewarded_terminal = 3
        self.random_sr = self.get_random_sr()

        self.reward_size = 1.
        self.trial_type = 'free'

    def toggle_training_trial(self):
        # block north arm
        self.states_actions_outcomes[2]['N'] = 2

        # unblock south arm
        self.states_actions_outcomes[2]['S'] = 1

        self.reward_size = 1.
        self.start_state = 1
        self.trial_type = 'train'
        self.landmark_location = self.get_state_location(4)

    def toggle_probe_trial(self):
        # block south arm
        self.states_actions_outcomes[2]['S'] = 2

        # unblock north arm
        self.states_actions_outcomes[2]['N'] = 4

        self.reward_size = 0.
        self.start_state = 4
        self.trial_type = 'probe'
        self.landmark_location = self.get_state_location(1)

    def reset(self):
        self.current_state = self.start_state

    def _fill_adjacency_matrix(self):
        self.adjacency_graph = np.zeros((self.nr_states, self.nr_states))

        for origin, outcome in self.states_actions_outcomes.items():
            for action, successor in outcome.items():
                self.adjacency_graph[origin, successor] = 1

    def act(self, action):
        """

        :param action:
        :return:
        """
        current_state = self.get_current_state()
        if self.reward_func is None and self.is_terminal(current_state):
            return 0
        else:
            next_state, reward = self.get_next_state_and_reward(current_state, action)
            self.current_state = next_state
            return next_state, reward

    def get_goal_state(self):
        return self.rewarded_terminal

    def get_orientation(self):
        pass

    def get_next_state_and_reward(self, origin, action):
        # If current state is terminal absorbing state:
        if self.is_terminal(origin):
            return origin, 0

        next_state = self.get_next_state(origin, action)
        reward = self.get_reward(origin, next_state)
        return next_state, reward

    def get_state_location(self, state_idx):
        return self.state_locs[state_idx]

    def get_state_idx(self, state_loc):
        for idx, loc in self.state_locs.items():
            if loc == (state_loc[0], state_loc[1]):
                return idx
        raise ValueError('Location does not correspond to state.')

    def get_next_state(self, origin, action):
        if type(action) == int or type(action) == np.int64:
            action = self.actions[action]
        return self.states_actions_outcomes[origin][action]

    def get_reward(self, origin, next_state):
        #if self.is_terminal(next_state) and origin == self.rewarded_terminal:
        if next_state == self.rewarded_terminal:
            return self.reward_size
        else:
            return 0.

    def get_feature_representation(self, state_idx):
        """Get one-hot feature representation from state index.
        """
        if self.is_terminal(state_idx):
            return np.zeros(self.nr_states)
        else:
            return np.eye(self.nr_states)[state_idx]

    def get_possible_actions(self, state_idx):
        return list(self.states_actions_outcomes[state_idx].keys())

    def get_transition_matrix(self, policy):
        transition_matrix = np.zeros([self.nr_states, self.nr_states])

        for state in self.state_indices:
            if self.is_terminal(state):
                continue
            for a, action in enumerate(self.actions):
                successor = self.states_actions_outcomes[state][action]
                transition_matrix[state, successor] += policy[state][a]
        return transition_matrix

    def get_successor_representation(self, policy, gamma=.95):
        transition_matrix = self.get_transition_matrix(policy)
        m = np.linalg.inv(np.eye(self.nr_states) - gamma * transition_matrix)
        return m

    def get_random_sr(self, gamma=.95):
        random_policy = generate_random_policy(self)
        random_walk_sr = self.get_successor_representation(random_policy, gamma=gamma)
        return random_walk_sr

    def show_maze(self, c_mappable=None, show_state_idx=True):
        fig, ax = plt.subplots(1)
        ax.set_aspect('equal')

        if c_mappable is not None:
            cmap = plt.get_cmap('inferno')
            c_mappable = c_mappable / c_mappable.max()

        for i, (x, y) in enumerate(self.state_locs.values()):
            if c_mappable is not None:
                colour = cmap(c_mappable[i])
            else:
                colour = 'gray'
            patch = Rectangle((x, y), width=1, height=1, edgecolor='k', color=colour)
            ax.add_patch(patch)

            if show_state_idx:
                ax.text(x + .5, y + .5, str(i), ha='center', va='center', size=10)

        plt.xlim([self.center[0] - self.max_dist / 2 - 1, self.center[0] + self.max_dist / 2 + 1])
        plt.ylim([self.center[1] - self.max_dist / 2 - 1, self.center[1] + self.max_dist / 2 + 1])
        ax.axis('off')


class TwoStepTask(Environment):

    data_folder = os.path.join(ROOT_FOLDER, 'data')

    def __init__(self, n_trials=272, common_prob=.7):
        super().__init__()
        self.n_trials = n_trials
        self.state_names = {0: 'initial',
                            1: 'left port',
                            2: 'right port',
                            3: 'left second',
                            4: 'right second',
                            5: 'terminal LL',
                            6: 'terminal LR',
                            7: 'terminal RL',
                            8: 'terminal RR'}
        self.state_names_lst = [val for val in self.state_names.values()]
        self.state_indices = [key for key in self.state_names.keys()]
        self.nr_states = len(self.state_names)
        self.nr_actions = 2
        self.common_prob = common_prob
        self.rare_prob = 1. - self.common_prob

        self.states_actions_outcomes = {
            'initial': {
                'L': ['left port'],
                'R': ['right port'],
            },
            'left port': {
                'P': ['left second', 'right second']
            },
            'right port': {
                'P': ['right second', 'left second']
            },
            'left second': {
                'L': ['terminal LL'],
                'R': ['terminal LR']
            },
            'right second': {
                'L': ['terminal RL'],
                'R': ['terminal RR']
            },
            'terminal LL': {},
            'terminal LR': {},
            'terminal RL': {},
            'terminal RR': {}
        }

        self.actions_per_state = {
            0: ['L', 'R'],
            1: ['P'],
            2: ['P'],
            3: ['L', 'R'],
            4: ['L', 'R'],
            5: [],
            6: [],
            7: [],
            8: []
        }

        self._fill_adjacency_matrix()
        self.set_transition_probabilities()
        self.create_graph()

        self.reward_traces = self.load_reward_traces()
        self.reward_probs = self.reward_traces[:, 0]

        self.start_state = 0
        self.current_state = self.start_state

        self.trial_count = 0

    def reset(self):
        self.current_state = self.start_state
        self.trial_count += 1
        self.reward_probs = self.reward_traces[:, self.trial_count]

    def _fill_adjacency_matrix(self):
        self.adjacency_graph = np.zeros((self.nr_states, self.nr_states))

        for idx in range(self.nr_states):
            state_name = self.state_names[idx]

            actions = self.states_actions_outcomes[state_name]
            for act, destination_list in actions.items():
                for dest in destination_list:
                    destination_idx = list(self.states_actions_outcomes.keys()).index(dest)
                    self.adjacency_graph[idx, destination_idx] = 1

    def set_transition_probabilities(self):
        """Set the transition probability matrix.
        """
        self.transition_probabilities = np.zeros((self.nr_states, self.nr_actions, self.nr_states))
        for state, act_outcome in self.states_actions_outcomes.items():
            s_idx = self.get_state_idx(state)
            for a_idx, (act, possible_destinations) in enumerate(act_outcome.items()):
                for i, d in enumerate(possible_destinations):
                    d_idx = self.get_state_idx(d)
                    if len(possible_destinations) == 1:
                        self.transition_probabilities[s_idx, a_idx, d_idx] = 1
                    elif len(possible_destinations) > 1 and i == 0:
                        self.transition_probabilities[s_idx, a_idx, d_idx] = self.common_prob
                    elif len(possible_destinations) > 1 and i == 1:
                        self.transition_probabilities[s_idx, a_idx, d_idx] = self.rare_prob

    def get_state_idx(self, state_name):
        return self.state_names_lst.index(state_name)

    def generate_reward_traces(self, **kwargs):
        """Generate reward traces per reward port per trial using a Gaussian random walk and save in file.
        :return:
        """
        r1 = self.bounded_random_walk(self.n_trials + 1, **kwargs)
        r2 = [1-r for r in r1]  # bounded_random_walk(self.n_trials, **kwargs)
        rewards = np.array([r1, r2, r1[::-1], r2[::-1]])
        if not os.path.isdir(self.data_folder):
            os.makedirs(self.data_folder)
        file_path = os.path.join(self.data_folder, 'reward_traces_anticorrelated.npy')
        np.save(file_path, rewards)

    def load_reward_traces(self):
        file_path = os.path.join(self.data_folder, 'reward_traces_anticorrelated.npy')
        try:
            reward_traces = np.load(file_path)
            tqdm.write('Loaded reward traces from file.')
        except FileNotFoundError:
            tqdm.write('Warning: No reward traces file was found so I generate a new one.')
            self.generate_reward_traces(avg_stepsize=.05, sigma=.0005)
            reward_traces = np.load(file_path)
        return reward_traces

    @staticmethod
    def bounded_random_walk(n_trials, lim=(.25, .75), avg_stepsize=.05, sigma=.005):
        rewards = [random.uniform(lim[0], lim[1])]
        for trial in range(n_trials - 1):

            stepsize = random.gauss(avg_stepsize, sigma)

            if rewards[trial] + stepsize > lim[1]:
                r = rewards[trial] - stepsize
            elif rewards[trial] + stepsize < lim[0]:
                r = rewards[trial] + stepsize
            elif random.random() >= .5:
                r = rewards[trial] + stepsize
            else:
                r = rewards[trial] - stepsize
            rewards.append(r)
        return rewards

    def is_terminal(self, state_idx):
        state_name = self.state_names[state_idx]
        return 'terminal' in state_name

    def get_state_name(self, idx):
        return self.state_names[idx]

    def get_action_idx(self, state, action):
        return list(self.states_actions_outcomes[state].keys()).index(action)

    def get_next_state(self, origin_idx, action_idx):
        origin_name = self.get_state_name(origin_idx)
        action_name = self.actions_per_state[origin_idx][action_idx]
        destination_idxs = [self.get_state_idx(d) for d in self.states_actions_outcomes[origin_name][action_name]]
        probabilities = [self.transition_probabilities[origin_idx, action_idx, d] for d in destination_idxs]
        next_state = np.random.choice(self.states_actions_outcomes[origin_name][action_name], p=probabilities)
        return self.state_names_lst.index(next_state)

    def get_possible_next_states(self, origin_idx, action_name):
        if self.is_terminal(origin_idx):
            return [origin_idx], [1]

        origin_name = self.get_state_name(origin_idx)
        action_idx = self.get_action_idx(origin_name, action_name)
        destination_idxs = [self.get_state_idx(d) for d in self.states_actions_outcomes[origin_name][action_name]]
        probabilities = [self.transition_probabilities[origin_idx, action_idx, d] for d in destination_idxs]
        next_states = self.states_actions_outcomes[origin_name][action_name]
        next_state_idxs = [self.get_state_idx(name) for name in next_states]
        return next_state_idxs, probabilities

    def get_reward(self, origin, destination):
        if not self.is_terminal(destination):
            return 0.
        if self.state_names[destination] == 'terminal LL':
            return np.random.choice([1., 0], p=[self.reward_probs[0], 1 - self.reward_probs[0]])
        elif self.state_names[destination] == 'terminal LR':
            return np.random.choice([1, 0], p=[self.reward_probs[1], 1 - self.reward_probs[1]])
        elif self.state_names[destination] == 'terminal RL':
            return np.random.choice([1, 0], p=[self.reward_probs[2], 1 - self.reward_probs[2]])
        elif self.state_names[destination] == 'terminal RR':
            return np.random.choice([1, 0], p=[self.reward_probs[3], 1 - self.reward_probs[3]])

    def get_next_state_and_reward(self, origin, action):
        if self.is_terminal(origin):
            return origin, 0

        next_state = self.get_next_state(origin, action)
        reward = self.get_reward(origin, next_state)
        return next_state, reward

    def act(self, action):
        current_state = self.get_current_state()
        if self.is_terminal(current_state):
            return 0
        else:
            next_state, reward = self.get_next_state_and_reward(current_state, action)
            self.current_state = next_state
            return next_state, reward

    def get_possible_actions(self, state_idx):
        state_name = self.state_names[state_idx]
        possible_actions = list(self.states_actions_outcomes[state_name].keys())
        return possible_actions

    def set_common_prob(self, prob):
        self.common_prob = prob
        self.rare_prob = 1 - prob
        self.set_transition_probabilities()


class DollTask(TwoStepTask):
    def __init__(self):
        super().__init__()

        self.state_names = {0: 'faces',
                            1: 'tools',
                            2: 'bodyparts',
                            3: 'scenes',
                            4: 'terminal LL',
                            5: 'terminal LR',
                            6: 'terminal RL',
                            7: 'terminal RR'}

        self.nr_states = len(self.state_names)
        self.state_indices = np.arange(self.nr_states)
        self.state_names_lst = [val for val in self.state_names.values()]

        self.actions_per_state = {
            0: ['L', 'R'],
            1: ['L', 'R'],
            2: ['L', 'R'],
            3: ['L', 'R'],
            4: [],
            5: [],
            6: [],
            7: []
        }

        self.states_actions_outcomes = {
            'faces': {
                'L': ['bodyparts'],
                'R': ['scenes']
            },
            'tools': {
                'L': ['bodyparts'],
                'R': ['scenes']
            },
            'bodyparts': {
                'L': ['terminal LL'],
                'R': ['terminal LR']
            },
            'scenes': {
                'L': ['terminal RL'],
                'R': ['terminal RR']
            },
            'terminal LL': {},
            'terminal LR': {},
            'terminal RL': {},
            'terminal RR': {}
        }

        self._fill_adjacency_matrix()
        self.set_transition_probabilities()
        self.create_graph()

    def reset(self):
        self.current_state = np.random.choice([0, 1])
        self.trial_count += 1
        self.reward_probs = self.reward_traces[:, self.trial_count]


class BlockingStudy(HexWaterMaze):
    bvc_maps_folder = os.path.join(ROOT_FOLDER, 'data', 'bvc_maps')

    def __init__(self, radius=6):
        super().__init__(radius=radius)
        self.scaling_factor = 750 / self.grid.radius
        self.landmark_2_location = self.set_landmark_2()
        self.corners = self.get_corners()
        self.boundaries = self.get_boundaries()
        self.all_boundaries = [deepcopy(b) for b in self.boundaries]
        self.n_boundaries = len(self.boundaries)
        self.boundaries_present = 'both'
        self.directions_radians = None

    def remove_left_boundary(self):
        self.restore_boundaries()
        if len(self.boundaries) < 6:
            raise ValueError('Cannot delete both boundaries')
        del self.boundaries[1:4]
        self.boundaries_present = 'right'
        self.n_boundaries = 3

    def remove_right_boundary(self):
        self.restore_boundaries()
        if len(self.boundaries) < 6:
            raise ValueError('Cannot delete both boundaries')
        del self.boundaries[-2:]
        del self.boundaries[0]
        self.boundaries_present = 'left'
        self.n_boundaries = 3

    def restore_boundaries(self):
        self.boundaries = [deepcopy(b) for b in self.all_boundaries]
        self.n_boundaries = 6

    def get_boundaries(self):
        corners_shifted = [self.corners[(i - 1) % len(self.corners)] for i, x in enumerate(self.corners)]

        boundaries = []
        for fro, to in zip(self.corners, corners_shifted):
            b = np.array((fro, to), dtype=np.float64)
            b = Boundary(b[0], b[1])
            boundaries.append(b)
        return boundaries

    def compute_bvc_maps(self):
        if not os.path.exists(self.bvc_maps_folder):
            os.makedirs(self.bvc_maps_folder)

        xs, ys = self.get_scaled_coords()

        distances = [81.0, 169.0, 265.0, 369.0, 482.5, 606.5, 800.]
        directions_degrees = np.linspace(0, 360, 61)[:-1]
        directions_radians = np.radians(directions_degrees)

        for dist in tqdm(distances, desc='Distances'):
            for orient in tqdm(directions_radians, leave=False, desc='orientations'):
                fname = 'bvcmap_{}_distance{}_orientation{}.npy'.format(self.boundaries_present,
                                                                        int(dist),
                                                                        int(np.round(np.degrees(orient))))
                if os.path.exists(os.path.join(self.bvc_maps_folder, fname)):
                    continue
                bvc = BVC(dist, orient)
                rate_map = bvc.compute_ratemap_grid(xs, ys, self)
                rate_map = rate_map / rate_map.max()
                np.save(os.path.join(self.bvc_maps_folder, fname), rate_map)

    def compute_bvc_maps_multiproc(self):
        if not os.path.exists(self.bvc_maps_folder):
            os.makedirs(self.bvc_maps_folder)

        distances = [81.0, 169.0, 265.0, 369.0, 482.5, 606.5, 800.]
        directions_degrees = np.linspace(0, 360, 61)[:-1]
        self.directions_radians = np.radians(directions_degrees)

        p = Pool(os.cpu_count() - 1)
        p.map(self.bvc_for_one_dist, distances)
        p.close()
        p.join()

    def bvc_for_one_dist(self, dist):
        xs, ys = self.get_scaled_coords()
        for orient in tqdm(self.directions_radians, leave=False, desc='orientations'):
            fname = 'bvcmap_{}_distance{}_orientation{}.npy'.format(self.boundaries_present,
                                                                    int(dist),
                                                                    int(np.round(np.degrees(orient))))
            if os.path.exists(os.path.join(self.bvc_maps_folder, fname)):
                continue
            bvc = BVC(dist, orient)
            rate_map = bvc.compute_ratemap_grid(xs, ys, self)
            rate_map = rate_map / rate_map.max()
            np.save(os.path.join(self.bvc_maps_folder, fname), rate_map)

    def get_scaled_coords(self):
        xs = []
        ys = []
        for s in range(self.nr_states):
            x, y = self.grid.cart_coords[s]
            xs.append(x * self.scaling_factor)
            ys.append(y * self.scaling_factor)
        return np.array(xs), np.array(ys)

    def get_corners(self):
        corners = []
        for angle in np.radians(self.allo_angles):
            point = (np.cos(angle) * self.grid.radius * self.scaling_factor,
                     np.sin(angle) * self.grid.radius * self.scaling_factor)
            corners.append(point)
        return corners

    def set_platform_state(self, state_idx):
        self.previous_platform_state = self.platform_state
        self.platform_state = state_idx
        self.landmark_location = self.set_landmark()
        self.set_reward_func()

    def set_landmark(self):
        platform_loc = self.grid.cube_coords[self.platform_state]
        landmark_loc = (platform_loc[0], platform_loc[1] + 2, platform_loc[2])
        return self.grid.to_cartesian(landmark_loc)

    def set_landmark_2(self):
        platform_loc = self.grid.cube_coords[self.platform_state]
        landmark_loc = (platform_loc[0], platform_loc[1] - 2, platform_loc[2])
        return self.grid.to_cartesian(landmark_loc)

    def remove_landmark_1(self):
        self.landmark_location = (100, 100)

    def remove_landmark_2(self):
        self.landmark_2_location = (100, 100)

    def toggle_learning_phase(self):
        self.landmark_location = self.set_landmark()
        self.remove_landmark_2()

    def toggle_compound_phase(self):
        self.landmark_location = self.set_landmark()
        self.landmark_2_location = self.set_landmark_2()

    def toggle_test_phase(self):
        self.remove_landmark_1()
        self.landmark_2_location = self.set_landmark_2()

    def reset(self, random_loc=False):
        if random_loc:
            dist_from_platform = np.array([self.grid.distance(self.platform_state, s) for s in self.grid.edge_states])
            eligible_start_states = np.array(self.grid.edge_states)[dist_from_platform > self.grid.radius]
            self.starting_state = np.random.choice(eligible_start_states)
        self.current_state = self.starting_state

    def draw_boundaries(self, ax, colors=None, linestyles=None):
        if colors is None:
            colors = 'k'
        if linestyles is None:
            linestyles = '-'
        lines = [(b.p1, b.p2) for b in self.boundaries]
        lc = mc.LineCollection(lines, colors=colors, linewidths=2, linestyles=linestyles)
        ax.add_collection(lc)
        ax.axis('equal')
        ax.autoscale()
        ax.axis('off')
        return ax


if __name__ == '__main__':


    en = TwoStepTask()

    pm = PlusMaze()
    pm.show_maze()
    plt.show()


    b = HexWaterMaze(5)
    randpol = generate_random_policy(b)
    m = b.get_successor_representation(randpol, gamma=.99)

    g = nx.from_numpy_array(b.adjacency_graph)
    nx.draw(g, node_color=m[52], pos=b.grid.cart_coords, with_labels=True)
    plt.show()

    b.plot_grid(m[0])
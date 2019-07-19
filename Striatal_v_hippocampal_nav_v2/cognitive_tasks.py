import numpy as np
import networkx as nx
import pandas as pd
import random
import os


class MDP(object):
    """General parent class for Markov Decision Processes.
    """
    def __init__(self):
        self.nr_states = None
        self.state_names = []
        self.actions = []
        self.nr_actions = None
        self.adjacency_graph = None
        self.goal_state = None
        self.reward_func = None
        self.graph = None
        self.n_features = None
        self.rf = None
        self.transition_probabilities = None
        self.terminal_state = None
        self.states_actions_outcomes = {}
        self.curr_state = None
        self.curr_action_idx = None
        self.reward_probs = None

    def create_graph(self):
        """Create networkx graph from adjacency matrix.
        """
        self.graph = nx.from_numpy_array(self.adjacency_graph)

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

    def is_terminal(self, state_idx):
        if not self.get_possible_actions(state_idx):
            return True
        else:
            return False

    def get_possible_actions(self, state_idx):
        pass

    def get_degree_mat(self):
        degree_mat = np.eye(self.nr_states)
        for state, degree in self.graph.degree:
            degree_mat[state, state] = degree
        return degree_mat

    def get_destination_state(self, current_state, current_action):
        transition_probabilities = self.transition_probabilities[current_state, current_action]
        return np.flatnonzero(transition_probabilities)

    def get_state_idx(self, state_name):
        return self.state_names.index(state_name)

    def get_state_name(self, idx):
        return self.state_names[idx]

    def get_action_idx(self, state, action):
        return list(self.states_actions_outcomes[state].keys()).index(action)

    def current_state_is_terminal(self):
        return self.is_terminal(self.curr_state)

    def get_next_state(self, origin_idx, action_name):
        origin_name = self.get_state_name(origin_idx)
        action_idx = self.get_action_idx(origin_name, action_name)
        destination_idxs = [self.get_state_idx(d) for d in self.states_actions_outcomes[origin_name][action_name]]
        probabilities = [self.transition_probabilities[origin_idx, action_idx, d] for d in destination_idxs]
        next_state = np.random.choice(self.states_actions_outcomes[origin_name][action_name], p=probabilities)
        return self.state_names.index(next_state)

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

    def get_next_state_by_name(self, origin, action):
        state_idx = self.get_state_idx(origin)
        action_idx = self.get_action_idx(origin, action)
        destination_idxs = [self.get_state_idx(d) for d in self.states_actions_outcomes[origin][action]]
        probabilities = [self.transition_probabilities[state_idx, action_idx, d] for d in destination_idxs]
        next_state = np.random.choice(self.states_actions_outcomes[origin][action], p=probabilities)
        return next_state

    def get_reward(self, origin, destination):
        if not self.is_terminal(destination):
            return 0
        if self.state_names[destination] == 'terminal1':
            return np.random.choice([1, 0], p=[self.reward_probs[0], 1 - self.reward_probs[0]])
        elif self.state_names[destination] == 'terminal2':
            return np.random.choice([1, 0], p=[self.reward_probs[1], 1 - self.reward_probs[1]])
        elif self.state_names[destination] == 'terminal3':
            return np.random.choice([1, 0], p=[self.reward_probs[2], 1 - self.reward_probs[2]])
        elif self.state_names[destination] == 'terminal4':
            return np.random.choice([1, 0], p=[self.reward_probs[3], 1 - self.reward_probs[3]])

    def get_next_state_and_reward(self, origin, action):
        if self.is_terminal(origin):
            return origin, 0

        next_state = self.get_next_state(origin, action)
        reward = self.get_reward(origin, next_state)
        return next_state, reward

    def act(self, action):
        self.curr_action_idx = np.flatnonzero(self.actions == action)[0]
        current_state = self.get_current_state()
        if self.is_terminal(current_state):
            return 0
        else:
            next_state, reward = self.get_next_state_and_reward(current_state, action)
            self.curr_state = next_state
            return next_state, reward

    def get_current_state(self):
        return self.curr_state


class DeterministicTask(MDP):
    """This class implements the deterministic two-step task described in Doll et al (2015, Nature Neuroscience).
    """
    output_folder = 'Data/DeterministicTask/'

    def __init__(self, n_trials=272):
        MDP.__init__(self)
        self.state_names = ['faces', 'tools', 'bodyparts', 'scenes',
                            'terminal1', 'terminal2', 'terminal3', 'terminal4']
        self.actions = np.array(['left', 'right'])
        self.nr_states = len(self.state_names)
        self.n_trials = n_trials
        self.nr_actions = 2
        self.states_actions_outcomes = {
            'faces': {
                'left': ['bodyparts'],
                'right': ['scenes']
            },
            'tools': {
                'left': ['bodyparts'],
                'right': ['scenes']
            },
            'bodyparts': {
                'left': ['terminal1'],
                'right': ['terminal2']
            },
            'scenes': {
                'left': ['terminal3'],
                'right': ['terminal4']
            },
            'terminal1': {},
            'terminal2': {},
            'terminal3': {},
            'terminal4': {}
        }

        self._fill_adjacency_matrix()
        self.set_transition_probabilities()
        self.create_graph()

        self.reward_traces = self.load_reward_traces()
        self.reward_probs = self.reward_traces[:, 0]  # [1, 0, .1, .1]

        self.start_state = 0  # Change this to stochastic 0 or 1
        self.curr_state = self.start_state
        self.curr_action_idx = 0

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

    def get_possible_actions(self, state_idx):
        state_name = self.state_names[state_idx]
        possible_actions = list(self.states_actions_outcomes[state_name].keys())
        return possible_actions

    def get_state_idx(self, state_name):
        return self.state_names.index(state_name)

    def is_terminal(self, state_idx):
        state_name = self.state_names[state_idx]
        return self.states_actions_outcomes[state_name] == {}

    def reset(self):
        self.start_state = np.random.choice([0, 1])
        self.curr_state = self.start_state

    def plot_graph(self, map_variable=None, node_size=1500, **kwargs):
        positions = {0: [0.25, 2], 1: [1.25, 2], 2: [.25, 1], 3: [1.25, 1],
                     4: [0, 0], 5: [.5, 0], 6: [1, 0], 7: [1.5, 0]}
        self.show_graph(map_variable=map_variable, node_size=node_size,
                        layout=positions, **kwargs)

    def generate_reward_traces(self, **kwargs):
        """Generate reward traces per reward port per trial using a Gaussian random walk and save in file.
        :return:
        """
        r1 = bounded_random_walk(self.n_trials, **kwargs)
        r2 = [1-r for r in r1]  # bounded_random_walk(self.n_trials, **kwargs)
        #r2 = bounded_random_walk(self.n_trials, **kwargs)
        rewards = np.array([r1, r2, r1[::-1], r2[::-1]])
        file_path = os.path.join(self.output_folder, 'reward_traces_anticorrelated.npy')
        np.save(file_path, rewards)

    def load_reward_traces(self):
        file_path = os.path.join(self.output_folder, 'reward_traces_anticorrelated.npy')
        try:
            reward_traces = np.load(file_path)
            print('Loaded reward traces from file.')
        except FileNotFoundError:
            print('Warning: No reward traces file was found so I generate a new one.')
            self.generate_reward_traces(avg_stepsize=.05, sigma=.0005)
            reward_traces = np.load(file_path)
        return reward_traces


class StochasticTask(MDP):
    """This class implements the stochastic two-step task as described in Daw et al. (2011, Neuron).
    """
    output_folder = 'Data/StochasticTask'

    def __init__(self, n_trials=272):
        MDP.__init__(self)
        self.state_names = ['initiation', 'left_state', 'right_state',
                            'terminal1', 'terminal2', 'terminal3', 'terminal4']
        self.common_probability = .7
        self.rare_probability = 1 - self.common_probability

        self.actions = np.array(['left', 'right'])
        self.nr_states = len(self.state_names)
        self.n_trials = n_trials
        self.nr_actions = 2
        self.states_actions_outcomes = {
            'initiation': {
                'left': ['left_state', 'right_state'],
                'right': ['right_state', 'left_state']
            },
            'left_state': {
                'left': ['terminal1'],
                'right': ['terminal2']
            },
            'right_state': {
                'left': ['terminal3'],
                'right': ['terminal4']
            },
            'terminal1': {},
            'terminal2': {},
            'terminal3': {},
            'terminal4': {}
        }

        self._fill_adjacency_matrix()
        self.set_transition_probabilities()
        self.create_graph()

        self.reward_traces = self.load_reward_traces()
        self.reward_probs = self.reward_traces[:, 0]  # [1, 0, .1, .1]

        self.start_state = 0
        self.curr_state = self.start_state
        self.curr_action_idx = 0

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
                        self.transition_probabilities[s_idx, a_idx, d_idx] = self.common_probability
                    elif len(possible_destinations) > 1 and i == 1:
                        self.transition_probabilities[s_idx, a_idx, d_idx] = self.rare_probability

    def plot_graph(self, map_variable=None, node_size=1500, **kwargs):
        positions = {0: [0.75, 2], 1: [.25, 1], 2: [1.25, 1],
                     3: [0, 0], 4: [.5, 0], 5: [1, 0], 6: [1.5, 0]}
        self.show_graph(map_variable=map_variable, node_size=node_size,
                        layout=positions, **kwargs)

    def generate_reward_traces(self, **kwargs):
        """Generate reward traces per reward port per trial using a Gaussian random walk and save in file.
        :return:
        """
        r1 = bounded_random_walk(self.n_trials, **kwargs)
        r2 = [1-r for r in r1]  # bounded_random_walk(self.n_trials, **kwargs)
        #r2 = bounded_random_walk(self.n_trials, **kwargs)
        rewards = np.array([r1, r2, r1[::-1], r2[::-1]])
        file_path = os.path.join(self.output_folder, 'reward_traces_anticorrelated.npy')
        np.save(file_path, rewards)

    def load_reward_traces(self):
        file_path = os.path.join(self.output_folder, 'reward_traces_anticorrelated.npy')
        try:
            reward_traces = np.load(file_path)
            print('Loaded reward traces from file.')
        except FileNotFoundError:
            print('Warning: No reward traces file was found so I generate a new one.')
            self.generate_reward_traces(avg_stepsize=.05, sigma=.0005)
            reward_traces = np.load(file_path)
        return reward_traces

    def reset(self):
        self.curr_state = self.start_state

    def get_possible_actions(self, state_idx):
        state_name = self.state_names[state_idx]
        possible_actions = list(self.states_actions_outcomes[state_name].keys())
        return possible_actions

    def get_state_idx(self, state_name):
        return self.state_names.index(state_name)

    def is_terminal(self, state_idx):
        state_name = self.state_names[state_idx]
        return self.states_actions_outcomes[state_name] == {}


def bounded_random_walk(n_trials, lim=(.25, .75), avg_stepsize=.05, sigma=.005):
    rewards = [random.uniform(lim[0], lim[1])]
    for trial in range(n_trials-1):

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


if __name__ =="__main__":

    b = StochasticTask()
    b.act('leftface')
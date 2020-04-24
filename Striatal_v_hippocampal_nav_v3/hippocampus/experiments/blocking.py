from hippocampus.environments import BlockingStudy, Environment
from hippocampus import utils
from hippocampus.utils import to_agent_frame
from hippocampus.dynamic_programming_utils import generate_random_policy, value_iteration
from hippocampus.agents import SRTD, LandmarkCells

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from multiprocessing import Pool
import os
from hippocampus.plotting import tsplot_boot
from definitions import ROOT_FOLDER


class SFTD(object):
    """Successor Features agent. Features are place cells with BVC inputs.
    """
    map_folder = os.path.join(ROOT_FOLDER, 'data', 'bvc_maps')

    def __init__(self, env=BlockingStudy(), beta=20, eta=.03, gamma=.99):
        self.env = env
        self.learning_rate = .05
        self.epsilon = .1
        self.gamma = gamma
        self.beta = beta
        self.eta = eta

        self.reliability = .8
        self.omega = 1.  # np.ones(self.env.nr_states)

        self.nr_features = 0
        self.features = self.load_features()
        random_policy = generate_random_policy(self.env)

        # note: SF works well with SR as features. not so well when gamma is too high, so features need to be zero when
        # farther away
        M_hat = self.env.get_successor_representation(random_policy, gamma=.9)

        #self.features = np.eye(self.env.nr_states)  #M_hat  #
        #self.features = M_hat
        #self.nr_features = self.env.nr_states

        # SR initialisation
        self.W_hat = self.init_M()

        self.identity = np.eye(self.nr_features)
        self.R_weights = np.zeros(self.nr_features)  # the reward expectation for each feature

    def load_features(self):
        fname = os.path.join(self.map_folder, 'place_cell_activity_{}_boundary.npy'.format(self.env.boundaries_present))
        features = np.load(fname)

        for i in features:
            i /= i.sum() / 10

        self.nr_features = features.shape[1] +1
        platform_feature = np.eye(self.env.nr_states)[self.env.platform_state]

        augmented_features = np.zeros((self.env.nr_states, self.nr_features))
        augmented_features[:, :-1] = features
        augmented_features[self.env.platform_state] = np.zeros(self.nr_features)
        augmented_features[:, -1] = platform_feature

        features = augmented_features
        return features

    def get_feature_rep(self, state):
        return self.features[state]

    def init_M(self, ):
        M_hat = np.eye(self.nr_features)
        return M_hat

    def get_SR(self, state):
        return self.W_hat.T @ self.get_feature_rep(state)

    def one_episode(self, random_policy=False):
        time_limit = 1000
        self.env.reset()
        t = 0
        s = self.env.get_current_state()
        cumulative_reward = 0

        results = pd.DataFrame({'time': [],
                                'reward': [],
                                'RPE': [],
                                'reliability': [],
                                'state': []})

        while not self.env.is_terminal(s) and t < time_limit:
            if random_policy:
                a = np.random.choice(list(range(self.env.nr_actions)))
            else:
                a = self.select_action(s)

            next_state, reward = self.env.act(a)

            SPE = self.compute_error(next_state, s)

            self.update_reliability(SPE, s)
            self.W_hat += self.update_M(SPE, s)
            self.update_R(next_state, reward)

            s = next_state
            t += 1
            cumulative_reward += reward

            results = results.append({'time': t, 'reward': reward, 'SPE': SPE, 'reliability': self.reliability,
                                      'state': s}, ignore_index=True)

        return results

    def update_R(self, next_state, reward):
        RPE = reward - np.dot(self.R_weights, self.get_feature_rep(next_state))
        self.R_weights += .1 * self.get_feature_rep(next_state) * RPE

    def update_M(self, SPE, s):
        delta_M = self.learning_rate * np.outer(self.get_feature_rep(s), SPE)
        return delta_M

    def update_reliability(self, SPE, s):
        max_feature = np.argmax(self.get_feature_rep(s))
        self.reliability += self.eta * (1 - abs(SPE[max_feature]) / 1 - self.reliability)

    def compute_error(self, next_state, s):
        if self.env.is_terminal(next_state):
            SPE = self.get_feature_rep(s) + self.get_feature_rep(next_state) - self.get_SR(s)
            #SPE = self.get_feature_rep(s) + self.identity[next_state] - self.get_SR(s)
        else:
            SPE = self.get_feature_rep(s) + self.gamma * self.get_SR(next_state) - self.get_SR(s)
        return SPE

    def select_action(self, state_idx, softmax=True):
        # TODO: get categorical dist over next state
        # okay because it's local
        # gradient-based planning (hill-climbing) gradient ascent
        # graph hill climbing
        # Maybe change for M(sa,sa). potentially over state action only in two step
        next_state = [self.env.get_next_state(state_idx, a) for a in range(self.env.nr_actions)]
        Q = [self.compute_V(s) for s in next_state]
        probabilities = utils.softmax(Q, self.beta)
        try:
            a = np.random.choice(list(range(self.env.nr_actions)), p=probabilities)
        except ValueError:
            print('whats wrong')
        return a

    def compute_V(self, s):
        return np.dot(self.get_SR(s), self.R_weights)


class LandmarkLearningAgent(object):
    """Q learning agent using landmark features.
    """
    max_RPE = 1

    def __init__(self, environment=BlockingStudy(6), learning_rate=.1, gamma=.9, eta=.03, beta=10):
        """

        :param environment:
        :param learning_rate:
        :param gamma:
        """
        self.env = environment
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.beta = beta
        self.eta = eta

        self.reliability = 0

        self.features = [LandmarkCells(), LandmarkCells()]
        self.n_cells = sum([l.n_cells for l in self.features])
        self.weights = np.zeros((self.n_cells, self.env.nr_actions))

    def one_episode(self, time_limit=1000):
        self.env.reset()

        t = 0
        cumulative_reward = 0
        s = self.env.get_current_state()
        orientation = 30  # np.random.choice([30, 90, 150, 210, 270, 330])
        f = self.get_feature_rep(s, orientation)
        Q = self.weights.T @ f

        results = pd.DataFrame({'time': [],
                                'reward': [],
                                'RPE': [],
                                'reliability': [],
                                'state': []})

        while not self.env.is_terminal(s) and t < time_limit:
            a = self.softmax_selection(s, Q)
            allo_a = self.get_allo_action(a, orientation)
            next_state, reward = self.env.act(allo_a)

            orientation = self.get_orientation(s, next_state, orientation)

            next_f = self.get_feature_rep(next_state, orientation)

            RPE, next_Q = self.compute_error(f, a, next_f, next_state, reward)

            if self.env.is_terminal(next_state):
                self.update_reliability(RPE)

            self.update_weights(RPE, a, f)

            cumulative_reward += reward
            s = next_state
            f = next_f
            Q = next_Q
            t += 1

            results = results.append({'time': t, 'reward': reward, 'RPE': RPE, 'reliability': self.reliability,
                                      'state': s}, ignore_index=True)
        return results

    def update_reliability(self, RPE):
        self.reliability += self.eta * ((1 - abs(RPE) / self.max_RPE) - self.reliability)

    def update_weights(self, RPE, a, f):
        self.weights[:, a] = self.weights[:, a] + self.learning_rate * RPE * f

    def compute_error(self, f, a, next_f, next_state, reward):
        Q = self.compute_Q(f)
        next_Q = self.weights.T @ next_f
        if self.env.is_terminal(next_state):
            RPE = reward - Q[a]
        else:
            RPE = reward + self.gamma * np.max(next_Q) - Q[a]
        return RPE, next_Q

    def softmax_selection(self, state_index, Q):
        probabilities = utils.softmax(Q, self.beta)
        action_idx = np.random.choice(list(range(self.env.nr_actions)), p=probabilities)
        return action_idx

    def angle_to_landmark(self, state, orientation):
        rel_pos = to_agent_frame(self.env.landmark_location, self.env.get_state_location(state), np.radians(orientation))
        angle = np.arctan2(rel_pos[1], rel_pos[0])
        return np.degrees(angle)

    def angle_to_landmark2(self, state, orientation):
        rel_pos = to_agent_frame(self.env.landmark_2_location, self.env.get_state_location(state), np.radians(orientation))
        angle = np.arctan2(rel_pos[1], rel_pos[0])
        return np.degrees(angle)

    def get_feature_rep(self, state, orientation):
        # for landmark 1
        distance = self.get_distance_to_landmark(state)
        angle = self.angle_to_landmark(state, orientation)
        response_to_l1 = self.features[0].compute_response(distance, angle)
        # for landmark 2
        distance = self.get_distance_to_landmark2(state)
        angle = self.angle_to_landmark2(state, orientation)
        response_to_l2 = self.features[1].compute_response(distance, angle)
        return np.concatenate([response_to_l1, response_to_l2])

    def get_distance_to_landmark(self, state):
        distance_to_landmark = np.linalg.norm(
            np.array(self.env.landmark_location) - np.array(self.env.get_state_location(state)))
        return distance_to_landmark

    def get_distance_to_landmark2(self, state):
        distance_to_landmark = np.linalg.norm(
            np.array(self.env.landmark_2_location) - np.array(self.env.get_state_location(state)))
        return distance_to_landmark

    def get_orientation(self, state, next_state, current_orientation):
        if state == next_state:
            return current_orientation
        s1 = self.env.get_state_location(state)
        s2 = self.env.get_state_location(next_state)
        return np.degrees(np.arctan2(s2[1] - s1[1], s2[0] - s1[0]))

    def get_allo_action(self, ego_action_idx, orientation):
        allo_angle = (orientation + self.env.ego_angles[ego_action_idx]) % 360
        for i, theta in enumerate(self.env.allo_angles):
            if theta == round(allo_angle):
                return i
        raise ValueError('Angle not in list.')

    def compute_Q(self, features):
        return self.weights.T @ features


class CombinedAgent(object):
    def __init__(self, env=BlockingStudy(6), init_sr='rw', lesion_dls=False, lesion_hpc=False, gamma=.95, eta=.03,
                 inv_temp=10, learning_rate=.1, inact_hpc=0., inact_dls=0.):
        self.inv_temp = inv_temp
        self.eta = eta
        self.learning_rate = learning_rate

        self.lesion_striatum = lesion_dls
        self.lesion_hippocampus = lesion_hpc
        if self.lesion_hippocampus and self.lesion_striatum:
            raise ValueError('cannot lesion both')
        self.env = env
        self.HPC = SFTD(self.env, gamma=gamma, eta=self.eta)
        self.DLS = LandmarkLearningAgent(self.env, eta=self.eta)
        self.current_choice = None

        self.weights = np.zeros((self.DLS.n_cells, self.env.nr_actions))
        self.gamma = gamma

        if inact_hpc:
            self.max_psr = 1. - inact_hpc
            self.p_sr = self.max_psr
            self.inact_dls = 0.
        elif inact_dls:
            self.max_psr = 1
            self.inact_dls = inact_dls
            self.p_sr = .8
        else:
            self.max_psr = 1
            self.inact_dls = 0.
            self.p_sr = .9

    def set_exploration(self, inv_temp):
        self.inv_temp = inv_temp

    def one_episode(self, random_policy=False, setp_sr=None):
        time_limit = 1000
        self.env.reset(random_loc=True)
        t = 0
        s = self.env.get_current_state()
        cumulative_reward = 0

        possible_orientations = np.round(np.degrees(self.env.action_directions))
        angles = []
        for i, o in enumerate(possible_orientations):
            angle = utils.angle_to_landmark(self.env.get_state_location(s), self.env.landmark_location, np.radians(o))
            angles.append(angle)
        orientation = possible_orientations[np.argmin(np.abs(angles))]

        # get MF system features
        f = self.DLS.get_feature_rep(s, orientation)
        Q_mf = self.weights.T @ f

        results = pd.DataFrame({})

        results = results.append({'time': t,
                                  'reward': 0,
                                  'SPE': 0,
                                  'RPE': 0,
                                  'HPC reliability': self.HPC.reliability,
                                  'DLS reliability': self.DLS.reliability,
                                  'alpha': self.get_alpha(self.DLS.reliability),
                                  'beta': self.get_beta(self.DLS.reliability),
                                  'state': s,
                                  'P(SR)': self.p_sr,
                                  'choice': self.current_choice,
                                  'M_hat': self.HPC.W_hat.flatten(),
                                  'R_hat': self.HPC.R_weights.copy(),
                                  'Q_mf': Q_mf,
                                  'platform': self.env.get_goal_state()}, ignore_index=True)

        while not self.env.is_terminal(s) and t < time_limit:
            if setp_sr is None:
                self.update_p_sr()
            else:
                self.p_sr = setp_sr

            # select action
            Q_combined, Q_allo = self.compute_Q(s, orientation, self.p_sr)
            if random_policy:
                allo_a = np.random.choice(list(range(self.env.nr_actions)))
            else:
                allo_a = self.softmax_selection(s, Q_combined)
            ego_a = self.get_ego_action(allo_a, orientation)

            # act
            next_state, reward = self.env.act(allo_a)

            # get MF state representation
            orientation = self.DLS.get_orientation(s, next_state, orientation)
            next_f = self.DLS.get_feature_rep(next_state, orientation)

            # SR updates
            SPE = self.HPC.compute_error(next_state, s)
            delta_M = self.HPC.learning_rate * SPE
            self.HPC.W_hat[s, :] += delta_M
            self.HPC.update_R(next_state, reward)

            # MF updates
            next_Q = self.weights.T @ next_f
            if self.env.is_terminal(next_state):
                RPE = reward - Q_mf[ego_a]
            else:
                RPE = reward + self.gamma * np.max(next_Q) - Q_mf[ego_a]

            self.weights[:, ego_a] = self.weights[:, ego_a] + self.learning_rate * RPE * f

            # Reliability updates
            if self.env.is_terminal(next_state):
                self.DLS.update_reliability(RPE)
                self.HPC.update_reliability(SPE, s)

            s = next_state
            f = next_f
            Q_mf = next_Q
            t += 1
            cumulative_reward += reward

            results = results.append({'time': t,
                                      'reward': reward,
                                      'SPE': SPE,
                                      'RPE': RPE,
                                      'HPC reliability': self.HPC.reliability,
                                      'DLS reliability': self.DLS.reliability,
                                      'alpha': self.get_alpha(self.DLS.reliability),
                                      'beta': self.get_beta(self.DLS.reliability),
                                      'state': s,
                                      'P(SR)': self.p_sr,
                                      'choice': self.current_choice,
                                      'M_hat': self.HPC.W_hat.copy(),
                                      'R_hat': self.HPC.R_weights.copy(),
                                      'Q_mf': Q_mf,
                                      'Q_allo': Q_allo,
                                      'Q': Q_combined,
                                      'features': f.copy(),
                                      'weights': self.weights.copy(),
                                      'platform': self.env.get_goal_state(),
                                      'landmark': self.env.landmark_location}, ignore_index=True)
        return results

    def get_ego_action(self, allo_a, orientation):
        ego_angle = round(utils.get_relative_angle(np.degrees(self.env.action_directions[allo_a]), orientation))
        if ego_angle == 180:
            ego_angle = -180
        for i, theta in enumerate(self.env.ego_angles):
            if theta == round(ego_angle):
                return i
        raise ValueError('Angle {} not in list.'.format(ego_angle))

    def update_p_sr(self):
        if self.lesion_hippocampus:
            self.p_sr = 0.
            return
        if self.lesion_striatum:
            self.p_sr = 1.
            return

        alpha = self.get_alpha(self.DLS.reliability)
        beta = self.get_beta(self.HPC.reliability)

        tau = self.max_psr / (alpha + beta)
        fixedpoint = (alpha + self.inact_dls * beta) * tau

        dpdt = (fixedpoint - self.p_sr) / tau

        new_p_sr = self.p_sr + dpdt
        if new_p_sr < 0 or new_p_sr > 1:
            raise ValueError('P(SR) is not a probability: {}'.format(new_p_sr))
        self.p_sr = new_p_sr

    @staticmethod
    def get_alpha(chi_mf):
        alpha1 = .01
        A = 1
        B = np.log((alpha1 ** -1) * A - 1)
        return A / (1 + np.exp(B * chi_mf))

    @staticmethod
    def get_beta(chi_mb):
        beta1 = .1
        A = .5
        B = np.log((beta1 ** -1) * A - 1)
        return A / (1 + np.exp(B * chi_mb))

    def compute_Q(self, state_idx, orientation, p_sr):

        # compute Q_SR
        V = self.HPC.W_hat @ self.HPC.R_weights
        next_state = [self.env.get_next_state(state_idx, a) for a in range(self.env.nr_actions)]
        Q_sr = [V[s] for s in next_state]

        # compute Q_MF
        features = self.DLS.get_feature_rep(state_idx, orientation)
        Q_ego = self.weights.T @ features

        allocentric_idx = [self.DLS.get_allo_action(idx, orientation) for idx in range(self.env.nr_actions)]

        Q_allo = np.empty(len(Q_ego))
        for i in range(len(Q_ego)):
            allo_idx = allocentric_idx[i]
            Q_allo[allo_idx] = Q_ego[i]

        Q_mf = Q_allo

        Q = p_sr * np.array(Q_sr) + (1-p_sr) * np.array(Q_mf)

        return Q, Q_mf

    def softmax_selection(self, state_index, Q):
        probabilities = utils.softmax(Q, self.inv_temp)
        action_idx = np.random.choice(list(range(self.env.nr_actions)), p=probabilities)
        return action_idx


if __name__ == "__main__":

    results_folder = os.path.join(ROOT_FOLDER, 'results', 'blocking')
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    n_agents = 100
    agent_indices = np.arange(n_agents)

    e = BlockingStudy(radius=6)
    e.set_platform_state(30)

    all_ets = []
    for i in tqdm(range(n_agents)):

        a = SFTD(e)
        escape_times = []

        e.remove_left_boundary()
        a.load_features()

        for trial in tqdm(range(30), leave=False):
            results = a.one_episode()
            et = results.time.max()
            escape_times.append(et)

        e.restore_boundaries()
        a.load_features()

        for trial in tqdm(range(30), leave=False):
            results = a.one_episode()
            et = results.time.max()
            escape_times.append(et)

        e.remove_left_boundary()
        a.load_features()

        for trial in tqdm(range(30), leave=False):
            results = a.one_episode()
            et = results.time.max()
            escape_times.append(et)

        all_ets.append(escape_times)


    results_boundary_blocking = np.array(all_ets)
    np.save(os.path.join(results_folder, 'boundary_blocking_results.npy'), results_boundary_blocking)

    fig, ax = plt.subplots()
    tsplot_boot(ax, np.array(all_ets))
    plt.show()

    #plt.plot(escape_times)
    #plt.show()


    def run_landmark_blocking(agent_idx):
        n_trials = 90

        np.random.seed(agent_idx)

        a = CombinedAgent(env=e, inv_temp=15, lesion_hpc=True)

        e.toggle_learning_phase()

        escapetimes = []
        for trial in tqdm(range(n_trials), leave=False):
            if trial == 30:
                e.toggle_compound_phase()
            if trial == 60:
                e.toggle_test_phase()
            results = a.one_episode()
            et = results.time.max()
            escapetimes.append(et)

        return np.array(escapetimes)

    p = Pool(os.cpu_count() - 1)

    landmark_blocking_results = p.map(run_landmark_blocking, agent_indices)
    p.close()
    p.join()

    landmark_blocking_results = np.array(landmark_blocking_results)
    np.save(os.path.join(results_folder, 'landmark_blocking_results.npy'), landmark_blocking_results)


    fig, ax = plt.subplots()
    tsplot_boot(ax, landmark_blocking_results)
    plt.show()

    #plt.plot(results)
    #plt.show()


from hippocampus.environments import TwoStepTask
from hippocampus.agents import LandmarkLearningAgent, QLearningTwoStep
from hippocampus import utils
from hippocampus.dynamic_programming_utils import generate_random_policy, value_iteration

import numpy as np
import pandas as pd


class SRTD(object):
    def __init__(self, env=TwoStepTask(), init_sr='identity', beta=20, eta=.03, gamma=.99):
        self.env = env
        self.learning_rate = .1
        self.epsilon = .1
        self.gamma = gamma
        self.beta = beta
        self.eta = eta

        self.reliability = 0.
        self.omega = 1.  # np.ones(self.env.nr_states)

        # SR initialisation
        self.M_hat = self.init_M(init_sr)

        self.identity = np.eye(self.env.nr_states)
        self.R_hat = np.zeros(self.env.nr_states)

    def init_M(self, init_sr):
        M_hat = np.zeros((self.env.nr_states, self.env.nr_states))
        if init_sr == 'zero':
            return M_hat
        if init_sr == 'identity':
            M_hat = np.eye(self.env.nr_states)
        elif init_sr == 'rw':  # Random walk initalisation
            random_policy = generate_random_policy(self.env)
            M_hat = self.env.get_successor_representation(random_policy, gamma=self.gamma)
        elif init_sr == 'opt':
            optimal_policy, _ = value_iteration(self.env)
            M_hat = self.env.get_successor_representation(optimal_policy, gamma=self.gamma)
        return M_hat

    def get_SR(self):
        return self.M_hat

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
            self.M_hat[s, :] += self.update_M(SPE)
            self.update_R(next_state, reward)

            s = next_state
            t += 1
            cumulative_reward += reward

            results = results.append({'time': t, 'reward': reward, 'SPE': SPE, 'reliability': self.reliability,
                                      'state': s}, ignore_index=True)

        return results

    def update_R(self, next_state, reward):
        RPE = reward - self.R_hat[next_state]
        self.R_hat[next_state] += 1. * RPE

    def update_M(self, SPE):
        delta_M = self.learning_rate * SPE
        return delta_M

    def update_reliability(self, SPE, s):
        self.reliability += self.eta * (1 - abs(SPE[s]) / 1 - self.reliability)

    def compute_error(self, next_state, s):
        if self.env.is_terminal(next_state):
            SPE = self.identity[s, :] + self.identity[next_state, :] - self.M_hat[s, :]
        else:
            SPE = self.identity[s, :] + self.gamma * self.M_hat[next_state, :] - self.M_hat[s, :]
        return SPE

    def select_action(self, state_idx, softmax=True):
        # TODO: get categorical dist over next state
        # okay because it's local
        # gradient-based (hill-climbing) gradient ascent
        # graph hill climbing
        # Maybe change for M(sa,sa). potentially over state action only in two step
        V = self.M_hat @ self.R_hat
        next_state = [self.env.get_next_state(state_idx, a) for a in range(self.env.nr_actions)]
        Q = [V[s] for s in next_state]
        probabilities = utils.softmax(Q, self.beta)
        return np.random.choice(list(range(self.env.nr_actions)), p=probabilities)


class CombinedAgent(object):
    def __init__(self, env=TwoStepTask(), init_sr='rw', lesion_dls=False, lesion_hpc=False, gamma=.95, eta=.03,
                 inv_temp=10, learning_rate=.2, lamb=.9, A_alpha=1., A_beta=.5, alpha1=.01, beta1=.1,
                 inact_hpc=0.):
        self.inv_temp = inv_temp
        self.eta = eta
        self.learning_rate = learning_rate
        self.lamb = lamb
        self.A_alpha = A_alpha
        self.A_beta = A_beta
        self.alpha1 = alpha1
        self.beta1 = beta1

        self.lesion_striatum = lesion_dls
        self.lesion_hippocampus = lesion_hpc
        if self.lesion_hippocampus and self.lesion_striatum:
            raise ValueError('cannot lesion both')
        self.env = env
        self.HPC = SRTD(self.env, init_sr=init_sr, gamma=gamma, eta=self.eta)
        self.DLS = QLearningTwoStep(self.env, eta=self.eta)
        self.current_choice = None

        self.weights = np.zeros((self.env.nr_states, self.env.nr_actions))
        self.trace = np.zeros(self.weights.shape)
        self.gamma = gamma

        self.p_sr = .1

        if inact_hpc:
            self.max_psr = 1. - inact_hpc
            self.p_sr = self.max_psr
            self.inact_dls = 0.
        else:
            self.max_psr = 1
            self.p_sr = .9

    def set_exploration(self, inv_temp):
        self.inv_temp = inv_temp

    def one_episode(self, random_policy=False, setp_sr=None):
        time_limit = 1000
        self.env.reset()
        t = 0
        s = self.env.get_current_state()
        states = [s]
        actions = []
        cumulative_reward = 0

        # get MF system features
        f = self.DLS.get_feature_rep(s, None)

        while not self.env.is_terminal(s) and t < time_limit:
            if setp_sr is None:
                self.update_p_sr()
            else:
                self.p_sr = setp_sr

            # select action
            Q_combined, Q_mf = self.compute_Q(s, None, self.p_sr)

            possible_actions = self.env.get_possible_actions(s)
            if random_policy:
                a = np.random.choice(list(range(len(possible_actions))))
            else:
                a = self.softmax_selection(s, Q_combined)

            actions.append(a)
            # act
            next_state, reward = self.env.act(a)

            # get MF state representation
            next_f = self.DLS.get_feature_rep(next_state, None)

            # SR updates
            SPE = self.HPC.compute_error(next_state, s)
            delta_M = self.HPC.learning_rate * SPE
            self.HPC.M_hat[s, :] += delta_M
            self.HPC.update_R(next_state, reward)

            # MF updates
            next_Q = self.weights.T @ next_f
            if self.env.is_terminal(next_state):
                RPE = reward - Q_mf[a]
            else:
                RPE = reward + self.gamma * np.max(next_Q) - Q_mf[a]

            F = np.zeros((len(f), 2))
            F[:, a] = f
            self.trace = F + self.lamb * self.trace
            self.weights = self.weights + self.learning_rate * RPE * self.trace

            # Reliability updates
            self.DLS.update_reliability(RPE)
            self.HPC.update_reliability(SPE, s)

            s = next_state
            states.append(s)
            f = next_f
            Q_mf = next_Q
            t += 1
            cumulative_reward += reward


        results = {'StartState': states[0],
                   'Action1': actions[0],
                   'Action2': actions[2],
                   'State2': states[2],
                   'Terminus': states[3],
                   'Reward': cumulative_reward,
                   'P(SR)': self.p_sr,
                   'HPC reliability': self.HPC.reliability,
                   'DLS reliability': self.DLS.reliability}
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
        fixedpoint = alpha * tau

        dpdt = (fixedpoint - self.p_sr) / tau

        new_p_sr = self.p_sr + dpdt

        if new_p_sr > self.max_psr:
            new_p_sr = self.max_psr
        if new_p_sr < 0:
            new_p_sr = 0
        if new_p_sr < 0 or new_p_sr > 1:
            raise ValueError('P(SR) is not a probability: {}'.format(new_p_sr))
        self.p_sr = new_p_sr

    def get_alpha(self, chi_mf):
        alpha1 = self.alpha1
        A = self.A_alpha
        B = np.log((alpha1 ** -1) * A - 1)
        return A / (1 + np.exp(B * chi_mf))

    def get_beta(self, chi_mb):
        beta1 = self.beta1
        A = self.A_beta
        B = np.log((beta1 ** -1) * A - 1)
        return A / (1 + np.exp(B * chi_mb))

    def compute_Q(self, state_idx, orientation, p_sr):

        possible_actions = self.env.get_possible_actions(state_idx)

        # compute Q_SR
        V = self.HPC.M_hat @ self.HPC.R_hat
        next_state = [self.env.get_next_state(state_idx, a) for a in range(len(possible_actions))]
        Q_sr = [V[s] for s in next_state]

        # compute Q_MF
        features = self.DLS.get_feature_rep(state_idx, orientation)
        Q_mf = self.weights.T @ features

        Q = p_sr * np.array(Q_sr) + (1-p_sr) * np.array(Q_mf)

        return Q, Q_mf

    def softmax_selection(self, state_index, Q):
        possible_actions = self.env.get_possible_actions(state_index)
        if len(possible_actions) == 1:
            return 0
        probabilities = utils.softmax(Q, self.inv_temp)
        action_idx = np.random.choice(list(range(len(possible_actions))), p=probabilities)
        return action_idx



if __name__ == '__main__':
    from tqdm import tqdm
    import os
    from definitions import RESULTS_FOLDER

    groups = ['control', 'lesion_hpc', 'lesion_dls']

    for group in tqdm(groups):

        if group == 'lesion_hpc':
            lesion_hpc = True
            lesion_dls = False
        elif group == 'lesion_dls':
            lesion_hpc = False
            lesion_dls = True
        else:
            lesion_dls = False
            lesion_hpc = False

        data_dir = os.path.join(RESULTS_FOLDER, 'twostep')
        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)

        df = pd.DataFrame({})

        n_agents = 100

        for agent in tqdm(range(n_agents), leave=False):
            e = TwoStepTask()
            a = CombinedAgent(env=e, lesion_dls=lesion_dls, lesion_hpc=lesion_hpc)

            for ep in tqdm(range(e.n_trials), leave=False):
                results = a.one_episode()
                results['Agent'] = agent
                results['Trial'] = ep
                df = df.append(results, ignore_index=True)

        df.to_csv(os.path.join(data_dir, 'results_{}.csv'.format(group)))
    tqdm.write('Done.')

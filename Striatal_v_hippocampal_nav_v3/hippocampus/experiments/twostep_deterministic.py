from hippocampus.experiments.twostep import CombinedAgent
from hippocampus.environments import DollTask

import pandas as pd
import numpy as np


class Agent(CombinedAgent):
    def __init__(self, env=DollTask(), lesion_dls=False, lesion_hpc=False):
        super().__init__(env=env, lesion_dls=lesion_dls, lesion_hpc=lesion_hpc, learning_rate=.2, inv_temp=6,
                         init_sr='identity')

    def one_episode(self, random_policy=False, setp_sr=None):
        time_limit = 1000
        self.env.reset()
        self.trace = np.zeros(self.trace.shape)
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
            if self.env.is_terminal(next_state):
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
                   'Action2': actions[1],
                   'State2': states[1],
                   'Terminus': states[-1],
                   'Reward': cumulative_reward,
                   'P(SR)': self.p_sr,
                   'HPC reliability': self.HPC.reliability,
                   'DLS reliability': self.DLS.reliability}
        return results


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

        data_dir = os.path.join(RESULTS_FOLDER, 'twostep_deterministic')
        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)

        df = pd.DataFrame({})

        n_agents = 100

        for agent in tqdm(range(n_agents), leave=False):
            e = DollTask()
            a = Agent(env=e, lesion_dls=lesion_dls, lesion_hpc=lesion_hpc)

            for ep in tqdm(range(e.n_trials), leave=False):
                results = a.one_episode()
                results['Agent'] = agent
                results['Trial'] = ep
                df = df.append(results, ignore_index=True)

        df.to_csv(os.path.join(data_dir, 'results_{}.csv'.format(group)))
    tqdm.write('Done.')

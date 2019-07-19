import numpy as np

from Experiments.Blocking.environment import Environment
from Experiments.Blocking.hippocampus_input import Hippocampus
from Experiments.Blocking.striatum import TDStriatum


class Agent(object):
    def __init__(self, env=Environment(), epsilon=.1, zeta=1.2, lesion_striatum=False, lesion_hippocampus=False):
        self.epsilon = epsilon
        self.zeta = zeta
        if lesion_hippocampus and lesion_striatum:
            self.epsilon = 1  # Output random behaviour if no striatum and no hippocampus are present.
        self.striatum_lesion = lesion_striatum
        self.hippocampus_lesion = lesion_hippocampus
        self.env = env
        self.hippocampus = Hippocampus(self.env)
        self.striatum = TDStriatum(self.env)
        self.reached_goal = False

    def train_one_episode(self):
        self.env.reset()
        self.reached_goal = False
        reached_terminus = False
        self.striatum.update()
        self.hippocampus.update()

        t = 0
        reward = 0

        locs = [[self.env.curr_x, self.env.curr_y]]
        choices = []

        while not self.reached_goal and not reached_terminus and t < self.env.timeout / self.env.time_bin:

            action, expert = self.choose_action()
            self.reached_goal = self.env.act(action)
            if self.reached_goal:
                reward = 1

            self.hippocampus.update()
            self.striatum.update()

            t += 1

            locs.append([self.env.curr_x, self.env.curr_y])
            choices.append(expert)

        return t, reward, np.array(locs), choices

    def choose_action(self):
        """Choose action from both hippocampus and striatum and compare their value.
        """
        if np.random.rand() <= self.epsilon:
            action = np.random.choice(self.env.actions)
            return action, 'random'

        hc_action, hc_value = self.hippocampus.choose_action()  # output direction in degrees.
        str_action, str_value = self.striatum.choose_action()

        if self.hippocampus_lesion:
            return str_action, 'striatum'
        elif self.striatum_lesion:
            return hc_action, 'hippocampus'

        if hc_value == str_value:
            choice = np.random.choice(['striatum', 'hippocampus'])
            if choice == 'hippocampus':
                return hc_action, 'hippocampus'
            else:
                return str_action, 'striatum'
        elif hc_value > self.zeta * str_value:
            return hc_action, 'hippocampus'
        else:
            return str_action, 'striatum'


def set_cues(cues, agent):
    assert cues == 'A' or 'B' or 'both', 'Wrong input argument, has to be A, B or both'
    agent.env.cues_present = cues
    for pc in agent.hippocampus.place_cells:
        pc.change_cues()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from plotting import tsplot_boot
    from tqdm import tqdm
    import os
    from definitions import ROOT_FOLDER
    from datetime import datetime

    n_simulations = 20
    n_episodes = 30

    envi = Environment()
    hippocampal_agents = [Agent(env=envi, lesion_striatum=True, lesion_hippocampus=False) for _ in range(n_simulations)]
    striatal_agents = [Agent(env=envi, lesion_hippocampus=True) for _ in range(n_simulations)]

    def run_simulation(agent):
        escape_times = np.empty(n_episodes)
        for ep in tqdm(range(n_episodes), desc='Episodes', leave=False):
            t, r, locs, choices = agent.train_one_episode()
            escape_times[ep] = t
        return escape_times


    random_seeds = np.arange(n_simulations)

    #results = p.map(run_simulation, agents)

    #results = [run_simulation(ag) for ag in tqdm(agents)]

    def run_blocking_experiment(agent, n_trials):
        escape_times = np.zeros((n_trials))
        for trial in range(n_trials):
            if trial ==8:
                print('something')
            if trial <= int(n_trials / 3):
                set_cues('A', agent)
            elif int(n_trials / 3) < trial < int(n_trials * 2 / 3):
                set_cues('both', agent)
            elif trial >= int(n_trials * 2 / 3):
                set_cues('B', agent)

            t, reward, locs, choices = agent.train_one_episode()
            escape_times[trial] = t
        return escape_times

    model_2_run = 'hippo'

    results_folder = os.path.join(ROOT_FOLDER, 'Data/Results/Blocking/')

    if model_2_run == 'hippo':
        results_hippo = np.array([run_blocking_experiment(ag, n_episodes) for ag in tqdm(hippocampal_agents)])
        filename_hippo = 'escape_times_hippo{}.npy'.format(datetime.now())
        np.save(os.path.join(results_folder, filename_hippo), results_hippo)
        results = results_hippo

    if model_2_run == 'striat':
        results_striat = np.array([run_blocking_experiment(ag, n_episodes) for ag in tqdm(striatal_agents)])
        filename_striatum = 'escape_times_striatum{}.npy'.format(datetime.now())
        np.save(os.path.join(results_folder, filename_striatum), results_striat)
        results = results_striat


    #results = p.map(run_blocking_experiment, zip(striatal_agents, np.tile(n_episodes, n_simulations), random_seeds))

    fig, ax = plt.subplots()
    tsplot_boot(ax, results)
    plt.show()

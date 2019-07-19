import numpy as np
from tqdm import tqdm
import os

from definitions import ROOT_FOLDER

from environments import WaterMazeEnv
from Experiments.WaterMaze.agent import TDAgent


data_folder = os.path.join(ROOT_FOLDER, 'Data/Results/WaterMaze')

red = '#f9868a'
blue = '#86c3f9'
green = '#86f9bc'
figure_folder = os.path.join(ROOT_FOLDER, 'Data/Results/WaterMaze/figures/')

if not os.path.isdir(figure_folder):
    os.makedirs(figure_folder)


def get_full_model_trajectories(n_episodes):
    all_choices = []
    all_locs = []
    a = TDAgent(env=WaterMazeEnv(), lesion_hippocampus=False, lesion_striatum=False)
    a.hippocampus.learning_rate = .04
    a.hippocampus.negative_learning_rate = .04
    a.hippocampus.lamb=.76
    a.env.start_x, a.env.start_y = [.4, 1.4]
    for ep in tqdm(range(n_episodes)):
        t, reward, locs, choices = a.train_one_episode()
        all_locs.append(locs)
        all_choices.append(choices)
    return np.array(all_choices), np.array(all_locs)


def get_agent_choices(n_agents=20, n_episodes=41):

    agent_choices = []

    for agent in tqdm(range(n_agents)):

        all_locs = []
        all_choices = []
        a = TDAgent(env=WaterMazeEnv(), lesion_hippocampus=False)

        a.hippocampus.learning_rate = .04
        a.hippocampus.negative_learning_rate = .04
        a.hippocampus.lamb=.8

        a.env.start_x, a.env.start_y = [.4, 1.4]
        for ep in range(n_episodes):
            t, reward, locs, choices = a.train_one_episode()
            all_locs.append(locs)
            all_choices.append(choices)

        agent_choices.append(all_choices)
    return np.array(agent_choices)


def evaluate_model_components(n_agents=5, n_episodes=35):
    all_rewards_hipp = np.zeros((n_agents, n_episodes))
    all_rewards_striat = np.zeros((n_agents, n_episodes))
    for ag in tqdm(range(n_agents), desc='Agent'):

        a = TDAgent(env=WaterMazeEnv(), lesion_hippocampus=True, lesion_striatum=False)
        a2 = TDAgent(env=WaterMazeEnv(), lesion_hippocampus=False, lesion_striatum=True)

        a2.hippocampus.learning_rate = .04
        a2.hippocampus.negative_learning_rate = .04
        a2.hippocampus.lamb = .7

        a.env.start_x, a.env.start_y = [.4, 1.4]
        a2.env.start_x, a2.env.start_y = [.4, 1.4]
        a2.env.curr_orientation = 0
        a.env.curr_orientation = 0

        for ep in tqdm(range(n_episodes), leave=False, desc='Trial'):
            t, reward, locs, choices = a.train_one_episode()
            t2, reward2, locs, choices = a2.train_one_episode()

            all_rewards_hipp[ag, ep] = t
            all_rewards_striat[ag, ep] = t2
    return all_rewards_hipp, all_rewards_striat


if __name__ == '__main__':

    n_episodes = 50

    filename = os.path.join(data_folder, 'example_trajectories.npz')
    if not os.path.isfile(filename):
        all_choices, all_locs = get_full_model_trajectories(n_episodes)
        np.savez(filename, name1=all_choices, name2=all_locs)

    filename = os.path.join(data_folder, 'example_trajectories.npz')
    if not os.path.isfile(filename):
        all_choices, all_locs = get_full_model_trajectories(n_episodes)
        np.savez(filename, name1=all_choices, name2=all_locs)

    fname = os.path.join(data_folder, 'escape_times.npz')
    if not os.path.isfile(fname):
        all_rewards_hipp, all_rewards_striat = evaluate_model_components(n_agents=20, n_episodes=35)
        np.savez(fname, name1=all_rewards_hipp, name2=all_rewards_striat)


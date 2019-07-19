import os
from functools import partial
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

from Experiments.PackardMcGaugh.agent import TDAgent as Agent
from combined_agent import Agent
from definitions import ROOT_FOLDER
from environments import PlusMaze

# Which brain area do we sample from?


n_sessions = 20
n_agents = 30


def run_session(agent, n_trials=3):
    """Run the Packard & McGaugh experiment.

    Agents run through the plus maze 3 times from one side and then perform an unrewarded probe trial starting from the
    other side.

    :param agent:
    :param n_trials:
    :return: The last location.
    """
    agent.env.start_on_original_side()
    for ep in range(n_trials):
        t, reward, _, choices = agent.train_one_episode()
    agent.env.start_on_opposite_side()
    t, reward, locs, choices = agent.train_one_episode()
    end_location = locs[-1]
    if end_location[0] <= .2:
        choice = 'place'
    elif end_location[0] >= .8:
        choice = 'response'
    else:
        choice = 'other'
    return choice


def _run_full_experiment(n_agents=30, n_sessions=10, lesion_hippocampus=False, lesion_striatum=False, **kwargs):
    """Run the full Packard & McGaugh experiment (naive for loop version).

    :param n_agents:
    :param n_sessions:
    :param lesion_hippocampus:
    :param lesion_striatum:
    :param kwargs:
    :return:
    """
    all_choices = []

    for ag in tqdm(range(n_agents)):
        agent = Agent(env=PlusMaze(), lesion_hippocampus=lesion_hippocampus, lesion_striatum=lesion_striatum, **kwargs)
        agent_choices = []
        for sess in range(n_sessions):
            choice = run_session(agent, n_trials=5)
            agent_choices.append(choice)
        all_choices.append(agent_choices)
    return np.array(all_choices)


def run_full_experiment(n_agents=30, n_sessions=10, lesion_hippocampus=False, lesion_striatum=False):
    """Run the full Packard & McGaugh experiment using multiprocessing.

    :param n_agents:
    :param n_sessions:
    :param lesion_hippocampus:
    :param lesion_striatum:
    :return:
    """
    p = Pool(n_agents)
    agent_indices = np.arange(n_agents)
    func = partial(run_one_experiment,
                   lesion_hippocampus=lesion_hippocampus,
                   lesion_striatum=lesion_striatum,
                   n_sessions=n_sessions)
    all_choices = p.map(func, agent_indices)

    p.close()
    p.join()
    return np.array(all_choices)


def run_one_experiment(agent_index, lesion_hippocampus, lesion_striatum, n_sessions):
    """Run the Packard & McGaugh experiment with a single agent.

    :param agent_index: To set different random seeds per process.
    :param lesion_hippocampus:
    :param lesion_striatum:
    :param n_sessions:
    :return:
    """
    np.random.seed(agent_index * 4)
    agent = Agent(env=PlusMaze(), lesion_hippocampus=lesion_hippocampus, lesion_striatum=lesion_striatum)
    agent_choices = []
    for sess in range(n_sessions):
        choice = run_session(agent, n_trials=5)
        agent_choices.append(choice)
    return agent_choices


if __name__ == '__main__':

    data_folder = os.path.join(ROOT_FOLDER, 'Data/Results/PackardMcGaugh')
    fname = 'PlusMazeData.npz'
    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    all_choices_control = run_full_experiment(n_agents=n_agents,
                                              n_sessions=n_sessions,
                                              lesion_striatum=False,
                                              lesion_hippocampus=False)

    all_choices_hipp_lesion = run_full_experiment(n_agents=n_agents,
                                                  n_sessions=n_sessions,
                                                  lesion_striatum=False,
                                                  lesion_hippocampus=True)

    all_choices_striat_lesion = run_full_experiment(n_agents=n_agents,
                                                    n_sessions=n_sessions,
                                                    lesion_striatum=True,
                                                    lesion_hippocampus=False)

    np.savez(os.path.join(data_folder, fname),
             control=all_choices_control,
             hipp_lesion=all_choices_hipp_lesion,
             striat_lesion=all_choices_striat_lesion)


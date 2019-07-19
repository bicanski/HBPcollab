import numpy as np
from environments import WaterMazeEnv
from tqdm import tqdm
import matplotlib.pyplot as plt
from plotting import tsplot_boot
from datetime import datetime
import os
from definitions import ROOT_FOLDER

from Experiments.Pearce.agent import TDAgent

output_folder = os.path.join(ROOT_FOLDER, 'Data/Results/Pearce')
if not os.path.isdir(output_folder):
    os.makedirs(output_folder)

#from combined_agent import TDAgent


def get_platform_and_landmark_locations(env, number=9):
    angles = np.linspace(0, 2*np.pi, number)
    r = env.maze_radius / 2
    platform_locations = [[r * np.cos(a), r * np.sin(a)] for a in angles]
    landmark_locations = [[r * np.cos(a), r * np.sin(a)+.1] for a in angles]
    platform_locations = env.maze_centre + platform_locations
    landmark_locations = env.maze_centre + landmark_locations
    return platform_locations, landmark_locations


envi = WaterMazeEnv()
platform_locations, landmark_locations = get_platform_and_landmark_locations(envi)

n_simulations = 10
n_episodes = 15


def run_landmark_displacement_experiment():
    global escape_times
    escape_times = np.zeros((n_simulations, n_episodes))
    for sim in tqdm(range(n_simulations)):
        ag = TDAgent(env=envi, lesion_striatum=False, lesion_hippocampus=False)
        ag.hippocampus.learning_rate = .04
        ag.hippocampus.negative_learning_rate = .05
        ag.striatum.learning_rate = .001
        ag.striatum.lamb = .6
        for ep in range(n_episodes):

            if ep == 0:
                envi.set_platform_and_landmark(platform_locations[0], landmark_locations[0])

            if ep == 5:
                envi.set_platform_and_landmark(platform_locations[4], landmark_locations[4])

            t, reward, locs, choices = ag.train_one_episode()
            escape_times[sim, ep] = t


#run_landmark_displacement_experiment()

#fig, ax = plt.subplots()
#tsplot_boot(ax, escape_times)

#plt.show()

# actual Pearce experiment

# Control:

n_sims = 50
n_trials = 4
n_sessions = 12

escape_times = np.zeros((n_sims, n_sessions, n_trials))
session_ids = list(range(len(platform_locations)))
sessions = [np.random.choice(session_ids) for _ in range(n_sessions)]

sessions = [0, 4, 1, 5, 2, 6, 3, 7, 4, 8, 3, 0]

first_trials_control_trajectory = []

sim = 0
for sim in tqdm(range(n_sims)):
    ag = TDAgent(env=envi, lesion_striatum=False, lesion_hippocampus=False)

    ag.hippocampus.max_goal_response = 20
    ag.striatum.lamb = .76
    ag.hippocampus.learning_rate = .04
    ag.hippocampus.negative_learning_rate = .04
    ag.striatum.learning_rate = .005

    for sess_n, session in enumerate(sessions):
        envi.set_platform_and_landmark(platform_locations[session], landmark_locations[session])
        for trial in range(n_trials):
            t, reward, locs, choices = ag.train_one_episode()
            if trial == 0:
                first_trials_control_trajectory.append(locs)
            escape_times[sim, sess_n, trial] = t


# Lesion

escape_times_lesion = np.zeros((n_sims, n_sessions, n_trials))
sim = 0

first_trials_lesion_trajectory = []

for sim in tqdm(range(n_sims)):
    ag = TDAgent(env=envi, lesion_hippocampus=True, epsilon=.1)

    ag.striatum.lamb = .76
    ag.hippocampus.learning_rate = .02
    ag.hippocampus.negative_learning_rate = .02
    ag.striatum.learning_rate = .005

    for sess_n, session in enumerate(sessions):
        envi.set_platform_and_landmark(platform_locations[session], landmark_locations[session])

        for trial in range(n_trials):
            t, reward, locs, choices = ag.train_one_episode()
            if trial == 0:
                first_trials_lesion_trajectory.append(locs)
            escape_times_lesion[sim, sess_n, trial] = t



time_step = 1

first_trials = escape_times[:, :, 0].mean(axis=0) * time_step
fourth_trials = escape_times[:, :, 3].mean(axis=0)* time_step

first_trials_lesion = escape_times_lesion[:, :, 0].mean(axis=0) * time_step
fourth_trials_lesion = escape_times_lesion[:, :, 3].mean(axis=0) * time_step

outfile = os.path.join(output_folder, 'pearce_results_{}'.format(datetime.now()))
np.savez(outfile, ctrl1=first_trials, ctrl4=fourth_trials,
         lesion1=first_trials_lesion, lesion4=fourth_trials_lesion)



fig, ax = plt.subplots(figsize=(5,5))
red = '#f9868a'
blue = '#86c3f9'


plt.plot(np.arange(1,12), first_trials_lesion[:-1], 'o-', color=red)
plt.plot(np.arange(1,12), first_trials[:-1], 'o-', fillstyle='none', color=blue)
plt.plot(np.arange(1,12), fourth_trials_lesion[:-1], 'o--', color=red)
plt.plot(np.arange(1,12), fourth_trials[:-1], 'o--', fillstyle='none', color=blue)

# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

#plt.ylim([0,122])
plt.ylabel('Escape latency (s)', fontsize=15)
plt.xlabel('Session', fontsize=15)

plt.legend(['Hippocampal lesion - trial 1', 'Control - trial 1',
            'Hippocampal lesion - trial 4', 'Control - trial 4'], fontsize=12)

plt.xticks(np.arange(0, 12))


#plt.savefig(os.path.join(output_path,'PearceModel-ClearLegend.svg'))

plt.show()
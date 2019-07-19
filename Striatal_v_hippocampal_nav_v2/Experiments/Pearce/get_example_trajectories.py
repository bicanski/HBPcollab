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
sessions = [0, 4, 1, 5, 2, 6, 3, 7, 4, 8, 3, 0]

n_sims = 1
n_trials = 4
n_sessions = 12

escape_times = np.zeros((n_sims, n_sessions, n_trials))
session_ids = list(range(len(platform_locations)))

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

    ag.epsilon = .13
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


def show_maze(ax, env, **kwargs):
    angles = np.linspace(0, 2 * np.pi, 100)
    x_marks = np.cos(angles) * env.maze_radius + env.maze_centre[0]
    y_marks = np.sin(angles) * env.maze_radius + env.maze_centre[1]
    ax.plot(x_marks, y_marks, color='k', LineWidth=3)


def plot_trajectories(ax, session_nr, all_paths, **kwargs):
    plt.sca(ax)
    ax.plot(all_paths[session_nr][:, 0], all_paths[session_nr][:, 1], **kwargs)
    platform_loc = platform_locations[sessions[session_nr]]
    previous_platform_loc = platform_locations[sessions[session_nr - 1]]

    platform = plt.Circle(platform_loc, envi.platform_radius, color='k', fill=None)
    ax.add_artist(platform)
    platform_previous = plt.Circle(previous_platform_loc, envi.platform_radius, linestyle='solid', fill=True,
                                   color='gray')
    ax.add_artist(platform_previous)

    show_maze(ax, envi, **kwargs)

    plt.box('off')
    plt.axis('equal')
    plt.xticks([])
    plt.yticks([])


filename = 'firsttrial_trajectorydata{}.npz'.format(datetime.now())
np.savez(os.path.join(output_folder, filename), ctrl=first_trials_control_trajectory,
         lesion=first_trials_lesion_trajectory)


fig, axs = plt.subplots(2,2)
plot_trajectories(axs[0,0], 7, first_trials_control_trajectory, color='blue')
plot_trajectories(axs[0,1], 7, first_trials_lesion_trajectory, color='red')
plot_trajectories(axs[1,0], 8, first_trials_control_trajectory, color='blue')
plot_trajectories(axs[1,1], 8, first_trials_lesion_trajectory, color='red')
plt.tight_layout(w_pad=-4, h_pad=4)
plt.show()

#plt.savefig(os.path.join(output_path,'PearceTrajectories.svg'))

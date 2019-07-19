import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import seaborn as sns
import os
from plotting import show_maze
from environments import WaterMazeEnv

from definitions import ROOT_FOLDER


results_directory = os.path.join(ROOT_FOLDER, 'Data/Results/Pearce')


def load_data(filename):
    data = np.load(os.path.join(results_directory, filename))
    return data


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


def plot_escape_times(ax, data, colour_palette):
    ax.plot(np.arange(1,12), data['ctrl1'][:-1], 'o-', color=colour_palette[1])
    ax.plot(np.arange(1,12), data['ctrl4'][:-1], 'o-', fillstyle='none', color=colour_palette[1])
    ax.plot(np.arange(1,12), data['lesion1'][:-1], 'o--', color=colour_palette[0])
    ax.plot(np.arange(1,12), data['lesion4'][:-1], 'o--', fillstyle='none', color=colour_palette[0])

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # plt.ylim([0,122])
    ax.set_ylabel('Escape latency (s)', fontsize=12)
    ax.set_xlabel('Session', fontsize=12)

    ax.legend(['Hippocampal lesion - trial 1', 'Control - trial 1',
                'Hippocampal lesion - trial 4', 'Control - trial 4'], fontsize=7)

    ax.set_xticks(np.arange(0, 12))


def plot_trajectories(ax, session_nr, all_paths, **kwargs):
    ax.plot(all_paths[session_nr][:, 0], all_paths[session_nr][:, 1], **kwargs)
    platform_loc = platform_locations[sessions[session_nr]]
    previous_platform_loc = platform_locations[sessions[session_nr - 1]]

    platform = plt.Circle(platform_loc, envi.platform_radius, color='k', fill=None)
    ax.add_artist(platform)
    platform_previous = plt.Circle(previous_platform_loc, envi.platform_radius, linestyle='solid', fill=True,
                                   color='gray')
    ax.add_artist(platform_previous)

    show_maze(ax, envi)

    ax.axis('off')
    ax.axis('equal')



def illustrate_task(ax, env):
    show_maze(ax, env)
    ax.axis('equal')
    ax.axis('off')

    for loc, ses in zip(platform_locations[:-1], sessions[:-1]):
        platform = plt.Circle(loc, env.platform_radius, color='k', fill=None)
        ax.add_artist(platform)
        ax.text(loc[0], loc[1]-.1, str(ses), ha='center', va='top', fontstyle='italic')

    for loc, ses in zip(landmark_locations[:-1], sessions[:-1]):
        ax.text(loc[0], loc[1], 'x', ha='center')


if __name__ == '__main__':

    filename = 'pearce_results_2019-02-12 15:15:45.795890.npz'
    data = load_data(filename)

    filename = 'firsttrial_trajectorydata2019-02-12 18:21:24.038530.npz'
    trajectory_data = load_data(filename)

    pastel_palette = sns.color_palette("Pastel1")
    myorder = [1, 4, 2, 0, 3, 5, 6, 7, 8]
    current_palette = [pastel_palette[i] for i in myorder]

    set3_palette = sns.color_palette("Set1")
    myorder = [3, 4, 2, 0, 1, 5, 6, 7, 8]
    current_palette = [set3_palette[i] for i in myorder]

    # make figure and gridspec
    fig = plt.figure(figsize=(8, 7))
    gs = GridSpec(10, 6)

    maze_ax = fig.add_subplot(gs[:4, :2])
    results_ax = fig.add_subplot(gs[:4, 4:])
    originaldata_ax = fig.add_subplot(gs[:4, 2:4])
    originaldata_ax.axis('off')
    originaltrajectory_ax = fig.add_subplot(gs[4:, :3])
    originaltrajectory_ax.axis('off')

    illustrate_task(maze_ax, WaterMazeEnv())

    # plot escape times
    plot_escape_times(results_ax, data, current_palette)

    # plot trajectories
    inner = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[5:, 3:],
                                    wspace=0.01, hspace=0.1)

    ax1 = plt.Subplot(fig, inner[0, 0])
    plot_trajectories(ax1, 7, trajectory_data['ctrl'], color=current_palette[1])
    fig.add_subplot(ax1)

    ax2 = plt.Subplot(fig, inner[0, 1])
    plot_trajectories(ax2, 7, trajectory_data['lesion'], color=current_palette[0])
    fig.add_subplot(ax2)

    ax3 = plt.Subplot(fig, inner[1, 0])
    plot_trajectories(ax3, 8, trajectory_data['ctrl'], color=current_palette[1])
    fig.add_subplot(ax3)

    ax4 = plt.Subplot(fig, inner[1, 1])
    plot_trajectories(ax4, 8, trajectory_data['lesion'], color=current_palette[0])
    fig.add_subplot(ax4)



    # Add figure labels
    maze_ax.text(-.1, 1.15, 'A', transform=maze_ax.transAxes,
             fontsize=16, fontweight='bold', va='top', ha='right')
    originaldata_ax.text(-.1, 1.15, 'B', transform=originaldata_ax.transAxes,
             fontsize=16, fontweight='bold', va='top', ha='right')
    originaltrajectory_ax.text(-.1, 1.06, 'C', transform=originaltrajectory_ax.transAxes,
             fontsize=16, fontweight='bold', va='top', ha='right')


    plt.tight_layout()
    plt.show()
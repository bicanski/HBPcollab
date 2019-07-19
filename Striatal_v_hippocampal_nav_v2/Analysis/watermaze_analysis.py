import os
from collections import Counter
from definitions import ROOT_FOLDER

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

from plotting import tsplot_boot, plot_trajectories
from environments import WaterMazeEnv


red = '#f9868a'
blue = '#86c3f9'
green = '#86f9bc'

data_folder = os.path.join(ROOT_FOLDER, 'Data/Results/WaterMaze/')
figure_folder = os.path.join(ROOT_FOLDER, 'Data/Results/WaterMaze/figures/')

if not os.path.isdir(figure_folder):
    os.makedirs(figure_folder)


def compute_proportions(agent_choices):
    n_agents, n_episodes = agent_choices.shape
    all_prop_striat = []
    all_prop_hipp = []
    for agent in range(n_agents):

        prop_striat = []
        prop_hipp = []

        for trial in range(n_episodes):
            c = Counter(agent_choices[agent][trial])
            prop_striat.append(c['striatum'] / sum(c.values()))
            prop_hipp.append(c['hippocampus'] / sum(c.values()))

        all_prop_striat.append(prop_striat)
        all_prop_hipp.append(prop_hipp)
    all_prop_striat = np.array(all_prop_striat)
    all_prop_hipp = np.array(all_prop_hipp)
    return all_prop_striat, all_prop_hipp


def plot_choice_proportions(ax, colour_palette):
    tsplot_boot(ax, all_prop_striat[:, :40], color=colour_palette[0])
    tsplot_boot(ax, all_prop_hipp[:, :40], color=colour_palette[1])
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('Trials', fontsize=12)
    ax.set_ylabel('Proportion of choices', fontsize=12)
    plt.savefig(os.path.join(figure_folder, 'ChoiceProportions.svg'))


filename = os.path.join(data_folder, 'example_trajectories.npz')
trajectorydata = np.load(filename)

all_choices = trajectorydata['name1']
all_locs = trajectorydata['name2']



# plot the proportions of choices
agent_choices = np.load(os.path.join(data_folder, 'agent_choices.npy'))
all_prop_striat, all_prop_hipp = compute_proportions(agent_choices)

fname = os.path.join(data_folder, 'escape_times.npz')
escapetime_data = np.load(fname)

all_rewards_hipp = escapetime_data['name1']
all_rewards_striat = escapetime_data['name2']


def plot_model_performance(ax, colour_palette):
    env = WaterMazeEnv()
    tsplot_boot(ax, all_rewards_striat[:, :35] * env.time_bin, color=colour_palette[0])
    tsplot_boot(ax, all_rewards_hipp[:, :35] * env.time_bin, color=colour_palette[1])

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend(['Striatum', 'Hippocampus'], fontsize=10)
    ax.set_xlabel('Trials', fontsize=12)
    ax.set_ylabel('Escape time (s)', fontsize=12)
    #ax.tight_layout()


if __name__ == '__main__':

    pastel_palette = sns.color_palette("Pastel1")
    myorder = [1, 4, 2, 0, 3, 5, 6, 7, 8]
    current_palette = [pastel_palette[i] for i in myorder]

    set3_palette = sns.color_palette("Set1")
    myorder = [3, 4, 2, 0, 1, 5, 6, 7, 8]
    current_palette = [set3_palette[i] for i in myorder]

    striatum_colour = current_palette[0]
    hipp_colour = current_palette[1]
    #plt.rcParams['font.family'] = 'sans-serif'  # maybe try helvetica on mac
    #plt.rcParams['font.sans-serif'] = 'Coolvetica'

    fig = plt.figure()
    gs = GridSpec(2, 4)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :2])
    ax4 = fig.add_subplot(gs[0:, 2:])

    early_trial = 3
    late_trial = 45

    plot_trajectories(ax1, WaterMazeEnv(), all_locs[early_trial], all_choices[early_trial], current_palette)
    plot_trajectories(ax2, WaterMazeEnv(), all_locs[late_trial], all_choices[late_trial], current_palette)
    plot_model_performance(ax3, colour_palette=current_palette)
    plot_choice_proportions(ax4, current_palette)

    ax1.text(.5, .8, 'Trial {}'.format(early_trial), transform=ax1.transAxes,
             fontsize=12, ha='center', style='italic')
    ax2.text(.5, .8, 'Trial {}'.format(late_trial), transform=ax2.transAxes,
             fontsize=12, ha='center', style='italic')


    ax1.text(-.1, 1.15, 'A', transform=ax1.transAxes,
             fontsize=16, fontweight='bold', va='top', ha='right')
    ax3.text(-.1, 1.15, 'B', transform=ax3.transAxes,
             fontsize=16, fontweight='bold', va='top', ha='right')
    ax4.text(-.1, 1.06, 'C', transform=ax4.transAxes,
             fontsize=16, fontweight='bold', va='top', ha='right')

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()

    plt.savefig(os.path.join(figure_folder, 'WaterMaze.png'))

    plt.show()

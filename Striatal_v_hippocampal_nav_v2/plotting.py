import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt, patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy import stats
import math


def tsplot(ax, data, **kw):
    """Time series plot to replace the deprecated seaborn function.

    :param ax:
    :param data:
    :param kw:
    :return:
    """
    x = np.arange(data.shape[1])
    est = np.mean(data, axis=0)
    sd = np.std(data, axis=0)
    cis = (est - sd, est + sd)
    ax.fill_between(x, cis[0], cis[1], alpha=0.2, **kw)
    ax.plot(x, est, **kw)
    ax.margins(x=0)


def tsplot_boot(ax, data, **kw):
    """Plot time series with bootstrapped confidence intervals.

    :param ax:
    :param data:
    :param kw:
    :return:
    """
    x = np.arange(data.shape[1])
    est = np.mean(data, axis=0)
    cis = _bootstrap(data)
    ax.fill_between(x, cis[0], cis[1], alpha=0.2, **kw)
    ax.plot(x, est, **kw)
    ax.margins(x=0)


def _bootstrap(data, n_boot=10000, ci=68):
    """Helper function for tsplot_boot. Bootstraps confidence intervals for plotting time series.

    :param data:
    :param n_boot:
    :param ci:
    :return:
    """
    boot_dist = []
    for i in range(int(n_boot)):
        resampler = np.random.randint(0, data.shape[0], data.shape[0])
        sample = data.take(resampler, axis=0)
        boot_dist.append(np.mean(sample, axis=0))
    b = np.array(boot_dist)
    s1 = np.apply_along_axis(stats.scoreatpercentile, 0, b, 50. - ci / 2.)
    s2 = np.apply_along_axis(stats.scoreatpercentile, 0, b, 50. + ci / 2.)
    return s1, s2


def plot_trace(agent, trials_to_plot=None):
    """Plot the swimming trajectory in the maze.
    """
    if not trials_to_plot:
        trials_to_plot = [1, int(agent.n_trials/2)+1, agent.n_trials]

    n_rows = int(math.ceil(len(trials_to_plot)/5))
    n_cols = int(math.ceil(len(trials_to_plot)/n_rows))
    fig, axs = plt.subplots(n_rows, n_cols, sharex='row', sharey='row')
    angles = np.linspace(0, 2 * np.pi, 100)

    x_marks = np.cos(angles) * agent.maze_radius + agent.maze_centre[0]
    y_marks = np.sin(angles) * agent.maze_radius + agent.maze_centre[1]

    axs = axs.ravel()

    for i, trial in enumerate(trials_to_plot):
        axs[i].plot(x_marks, y_marks)  # Draw the boundary of the circular maze
        trial_trajectory = agent.position_log[agent.position_log['Trial'] == trial]
        axs[i].plot(trial_trajectory['X position'], trial_trajectory['Y position'])
        axs[i].axis('equal')  # enforces equal axis sizes
        axs[i].set_title('Trial {}'.format(trial))

        platform = plt.Circle(agent.platform_centre, agent.platform_radius, color='g')
        axs[i].add_artist(platform)

        landmark1 = plt.Circle(agent.landmark_1_centre, agent.landmark_1_radius, color='r')
        axs[i].add_artist(landmark1)
        landmark2 = plt.Circle(agent.landmark_2_centre, agent.landmark_1_radius, color='y')
        axs[i].add_artist(landmark2)

        plt.xlim((agent.minx, agent.maxx))
        plt.ylim((agent.miny, agent.maxy))
        axs[i].tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off')
    return fig, axs


def show_landmark_and_platform(ax, env):
    platform = plt.Circle(env.platform_centre, env.platform_radius, color='g')
    ax.add_artist(platform)

    landmark1 = plt.Circle(env.landmark_1_centre, env.landmark_1_radius, color='r')
    ax.add_artist(landmark1)

    angles = np.linspace(0, 2 * np.pi, 100)
    x_marks = np.cos(angles) * env.maze_radius + env.maze_centre[0]
    y_marks = np.sin(angles) * env.maze_radius + env.maze_centre[1]
    ax.plot(x_marks, y_marks, color='k')


def uniqueish_color(choice, colour_1, colour_2):
    if choice == 'hippocampus':
        return colour_2
    elif choice == 'striatum':
        return colour_1
    else:
        return 'gray'


def plot_trajectories(ax, env, all_paths, choices, colour_palette, **kwargs):
    plt.sca(ax)

    i = 0
    for start, stop in zip(all_paths[:-1], all_paths[1:]):
        x, y = zip(start, stop)
        choice = choices[i]
        ax.plot(x, y, color=uniqueish_color(choice, colour_1=colour_palette[0], colour_2=colour_palette[1]))
        i += 1

    # ax.plot(all_paths[:,0], all_paths[:,1], **kwargs)
    platform_loc = env.platform_centre
    # previous_platform_loc =  platform_locations[sessions[session_nr-1]]

    platform = plt.Circle(platform_loc, env.platform_radius, color='k', fill=None)
    ax.add_artist(platform)

    show_maze(ax, env, **kwargs)

    plt.box(False)
    plt.axis('equal')
    plt.xticks([])
    plt.yticks([])


def show_maze(ax, env, **kwargs):
    angles = np.linspace(0, 2 * np.pi, 100)
    x_marks = np.cos(angles) * env.maze_radius + env.maze_centre[0]
    y_marks = np.sin(angles) * env.maze_radius + env.maze_centre[1]
    ax.plot(x_marks, y_marks, color='k', **kwargs)


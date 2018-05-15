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


class OptionPlotter(object):
    """Class containing methods to plot laplacian eigenvectors and the extracted options.
    """
    def __init__(self, environment):
        self.env = environment

    def plot_policy(self, policy):
        for idx in range(len(policy)):
            x, y = self.env.get_state_position(idx)

            action = policy[idx]

            dx = 0
            dy = 0

            if action == 0:  # down
                dy = .0001
            elif action == 1:  # right
                dx = .0001
            elif action == 2:  # up
                dy = -.0001
            elif action == 3:  # left
                dx = -.0001
            elif self.env.matrix_MDP[x][y] != -1 and action == 4:  # termination
                circle = plt.Circle(
                    (y + 0.5, self.env.num_rows - x + 0.5 - 1), 0.1, color='r')
                plt.gca().add_artist(circle)

            if self.env.matrix_MDP[x][y] != -1 and action != 4:
                plt.arrow(y + 0.5, self.env.num_rows - x + 0.5 - 1, dx, dy,
                      head_width=0.4, head_length=.4, fc='k', ec='k')
            elif self.env.matrix_MDP[x][y] == -1:
                plt.gca().add_patch(
                    patches.Rectangle(
                        (y, self.env.num_rows - x - 1),  # (x,y)
                        1.0,  # width
                        1.0,  # height
                        facecolor="black"
                    )
                )
            else:
                pass

        plt.xlim([0, self.env.num_cols])
        plt.ylim([0, self.env.num_rows])

        for i in range(self.env.num_cols):
            plt.axvline(i, color='k', linestyle=':')
        plt.axvline(self.env.num_rows, color='k', linestyle=':')

        for j in range(self.env.num_rows):
            plt.axhline(j, color='k', linestyle=':')
        plt.axhline(self.env.num_rows, color='k', linestyle=':')

    def plot_eigenvector(self, eigenvector, fig, ax):
        plt.imshow(eigenvector.reshape(self.env.num_rows, self.env.num_cols), cmap='coolwarm')

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(cax=cax)

    def plot_policy_and_eigenvector(self, policy, eigenvector, eigenvalue, number):
        fig, axs = plt.subplots(1, 2)

        plt.sca(axs[0])
        self.plot_policy(policy)
        aspect_ratio = np.diff(axs[0].get_xlim())[0] / np.diff(axs[0].get_ylim())[0]
        axs[0].set_aspect(aspect_ratio)

        plt.sca(axs[1])
        self.plot_eigenvector(eigenvector, fig, axs[1])
        axs[1].set_aspect(aspect_ratio)
        plt.suptitle('Option #{}: eigenvalue'.format(number + 1) + ' {}'.format(eigenvalue))
        plt.tight_layout()


import matplotlib.pyplot as plt
from Experiments.PackardMcGaugh.packard_mcgaugh_experiment import n_sessions, n_agents
from collections import Counter
import numpy as np
from definitions import ROOT_FOLDER
import os
from utils import zip_lists
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


data_folder = os.path.join(ROOT_FOLDER, 'Data/Results/PackardMcGaugh')
fname = 'PlusMazeData.npz'
choice_data = np.load(os.path.join(data_folder, fname))
all_choices_control = choice_data['control']
all_choices_str_lesion = choice_data['striat_lesion']
all_choices_hpc_lesion = choice_data['hipp_lesion']


def get_choice_proportions(choices):
    data = []
    for session in range(n_sessions):
        c = Counter(choices[:, session])
        choice_proportions = np.array([c['place'], c['response']]) / (c['place'] + c['response']) * 100
        data.append(choice_proportions)
    return np.array(data)


def plot_choice_proportions():
    r = np.arange(len(data_ctrl[:, 0]))
    plt.figure()
    bar_width = 0.85
    # plot
    names = r + 1
    # Create green Bars
    plt.bar(r, data_ctrl[:, 0], color='#b5ffb9', edgecolor='white', width=bar_width)
    # Create orange Bars
    plt.bar(r, data_ctrl[:, 1], bottom=data_ctrl[:, 0], color='#f9bc86', edgecolor='white', width=bar_width)
    # Custom x axis
    plt.xticks(r, names)
    plt.xlabel("Session")
    plt.ylabel('% choices')
    plt.legend(['Place', 'Response'])
    plt.title('Frequency of choices on probe trial')
    # Show graphic
    plt.show()


if __name__ == '__main__':
    data_ctrl = get_choice_proportions(all_choices_control)
    data_strl = get_choice_proportions(all_choices_str_lesion)
    data_hpcl = get_choice_proportions(all_choices_hpc_lesion)

    #plot_choice_proportions()

    bottom_saline = [data_ctrl[0][0], data_ctrl[0][0]]
    top_saline = [data_ctrl[0][1], data_ctrl[0][1]]
    bottom_lido = [data_strl[0][0], data_hpcl[0][0]]
    top_lido = [data_strl[0][1], data_hpcl[0][1]]

    bottom_saline_l = [data_ctrl[-1][0], data_ctrl[-1][0]]
    top_saline_l = [data_ctrl[-1][1], data_ctrl[-1][1]]
    bottom_lido_l = [data_strl[-1][0], data_hpcl[-1][0]]
    top_lido_l = [data_strl[-1][1], data_hpcl[-1][1]]

    barWidth = 0.35
    r1 = np.arange(len(bottom_saline))
    r2 = r1 + barWidth

    fig, axs = plt.subplots(1, 2, sharey=True, figsize=(7, 5))

    # make figure and gridspec
    #fig = plt.figure(figsize=(8, 7))
    #gs = GridSpec(2, 2)



    ## Early
    plt.sca(axs[0])
    plt.title('Early in training', y=1.15)
    # Saline group
    plt.bar(r1, bottom_saline, color='#b5ffb9', edgecolor='white', width=barWidth)
    plt.bar(r1, top_saline, bottom=bottom_saline, color='#f9bc86', edgecolor='white', width=barWidth)
    plt.box(False)
    # Lidocaine group
    plt.bar(r2, bottom_lido, color='#b5ffb9', edgecolor='white', width=barWidth)
    plt.bar(r2, top_lido, bottom=bottom_lido, color='#f9bc86', edgecolor='white', width=barWidth)
    plt.ylabel('% of agents')
    plt.xticks(zip_lists(r1, r2),
               [r'$\bf{Saline}$ ' + '\n(Control)', r'$\bf{Lidocaine}$ ' + '\n(Inactivation)',
                r'$\bf{Saline}$ ' + '\n(Control)', r'$\bf{Lidocaine}$ ' + '\n(Inactivation)'],
               rotation=60, fontsize=10)

    plt.tight_layout()
    plt.text(-.15, 105, 'Striatum')
    plt.text(.7, 105, 'Hippocampus')

    ## Late
    plt.sca(axs[1])
    plt.title('Late in training', y=1.15)

    # Saline group
    plt.bar(r1, bottom_saline_l, color='#b5ffb9', edgecolor='white', width=barWidth)
    plt.bar(r1, top_saline_l, bottom=bottom_saline_l, color='#f9bc86', edgecolor='white', width=barWidth)
    plt.box(False)
    # Lidocaine group
    plt.bar(r2, bottom_lido_l, color='#b5ffb9', edgecolor='white', width=barWidth)
    plt.bar(r2, top_lido_l, bottom=bottom_lido_l, color='#f9bc86', edgecolor='white', width=barWidth)

    plt.xticks(zip_lists(r1, r2),
               [r'$\bf{Saline}$ ' + '\n(Control)', r'$\bf{Lidocaine}$ ' + '\n(Inactivation)',
                r'$\bf{Saline}$ ' + '\n(Control)', r'$\bf{Lidocaine}$ ' + '\n(Inactivation)'],
               rotation=60, fontsize=10)
    plt.tight_layout(pad=5, w_pad=4, h_pad=3)
    # plt.text(-.2,-35,'Striatum')
    plt.text(-0.15, 105, 'Striatum')
    plt.text(.7, 105, 'Hippocampus')

    plt.legend(['Place Strategy', 'Response Strategy'])

    plt.show()

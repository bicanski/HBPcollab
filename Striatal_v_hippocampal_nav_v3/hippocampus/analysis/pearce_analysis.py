import os
import os.path as op
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from definitions import PEARCE_RESULTS_DIR, FIGURE_FOLDER

data_directories = {
    'lesion': op.join(PEARCE_RESULTS_DIR, 'lesion', '2019-11-06 01:22:41.370359'),  # 2019-11-05 10:46:28.222103'),
    'control': op.join(PEARCE_RESULTS_DIR, 'control', '2019-11-05 00:39:03.764484')  # '2019-11-05 00:39:03.764484')
}
figure_location = op.join(FIGURE_FOLDER, 'pearce')


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def create_summary_file(results_folder, n_agents=None):
    if n_agents is None:
        params = pd.read_csv(op.join(results_folder, 'params.csv'))
        n_agents = params['n_agents'].iloc[0]
    else:
        n_agents = n_agents

    all_data = []
    for ag in tqdm(range(n_agents), desc='loading data...', leave=False):
        df = pd.read_csv(op.join(results_folder, 'agent{}.csv'.format(ag)))
        summary = df.pivot_table(index=['agent', 'session', 'trial'], aggfunc='mean')
        all_data.append(summary)

    df = pd.concat(all_data)
    df['platform location'] = df['platform'].astype('category')
    df.to_csv(op.join(results_folder, 'summary.csv'))


def create_summaries():
    control_folder = op.join(PEARCE_RESULTS_DIR, 'control')
    lesion_folder = op.join(PEARCE_RESULTS_DIR, 'lesion')
    all_subs = get_immediate_subdirectories(control_folder)
    all_subs_l = get_immediate_subdirectories(lesion_folder)
    all_dirs = [op.join(control_folder, d) for d in all_subs] + [op.join(lesion_folder, d) for d in all_subs_l]

    for d in tqdm(all_dirs):
        if not op.isfile(op.join(d, 'summary.csv')):
            create_summary_file(d)
        else:
            tqdm.write('file already exists')


def load_summary(group):
    directory = data_directories[group]
    df = pd.read_csv(op.join(directory, 'summary.csv'))
    df['group'] = group
    df['Trial'] = df['trial'].astype('str')
    return df


def plot_escape_time():
    color_pal = sns.color_palette()

    df_l = load_summary('lesion')
    df_c = load_summary('control')

    agents_l = df_l.agent.unique()
    agents_c = df_c.agent.unique()

    df_l = df_l.set_index(['agent', 'session', 'trial'])
    df_c = df_c.set_index(['agent', 'session', 'trial'])

    first_l = df_l.loc[(list(agents_l), list(range(11)), 0)]
    last_l = df_l.loc[(list(agents_l), list(range(11)), 3)]
    first_c = df_c.loc[(list(agents_c), list(range(11)), 0)]
    last_c = df_c.loc[(list(agents_c), list(range(11)), 3)]

    # plot the escape time per session for trials 1 and 4
    fig, ax = plt.subplots()
    sns.lineplot(ax=ax, data=first_l.reset_index(), x='session', y='escape time', ci=None, c=color_pal[3], linewidth=3)
    sns.lineplot(ax=ax, data=last_l.reset_index(), x='session', y='escape time', ci=None, c=color_pal[3], linewidth=3)

    sns.lineplot(ax=ax, data=first_c.reset_index(), x='session', y='escape time', ci=None, c=color_pal[0], linewidth=3)
    sns.lineplot(ax=ax, data=last_c.reset_index(), x='session', y='escape time', ci=None, c=color_pal[0], linewidth=3)

    ax.lines[1].set_linestyle("--")
    ax.lines[3].set_linestyle("--")

    plt.legend(['HPC lesion - trial 1',
                'HPC lesion - trial 4',
                'Control - trial 1',
                'Control - trial 4'])

    if not op.exists(figure_location):
        os.makedirs(figure_location)
    plt.savefig(os.path.join(figure_location, 'pearce_escapetime_firstlast'), format='pdf')
    plt.show()
    plt.close()


if __name__ == '__main__':
    from hippocampus.environments import HexWaterMaze

    # create_summary_file(data_directories['lesion'])
    plot_escape_time()

    maze = HexWaterMaze(10)
    # pick example agent

    for group in ['control', 'lesion']:

        agents = {'control': 89, 'lesion': 12}

        session = 6

        df = pd.read_csv(op.join(data_directories[group], 'agent{}.csv'.format(agents[group])))

        s6t0 = df[np.logical_and(df.trial == 0, df.session == session)]
        previous_platform = df[df.session == session - 1].platform.iloc[0]
        s6t0['previous platform'] = previous_platform
        current_platform = s6t0.platform.iloc[0]
        maze.plot_occupancy_on_grid(s6t0, alpha=1., show_state_idx=False)

        plt.savefig(op.join(figure_location, 'pearce_occupancy_{}_agent{}_session{}.pdf'.format(group,
                                                                                                agents[group],
                                                                                                session)), format='pdf')
        plt.show()

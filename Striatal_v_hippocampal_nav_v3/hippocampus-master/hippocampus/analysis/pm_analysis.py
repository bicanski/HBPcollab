import pandas as pd
import os.path as op
import os
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

from definitions import RESULTS_FOLDER, FIGURE_FOLDER, ROOT_FOLDER

col_pal = sns.color_palette()

packard_folder = op.join(ROOT_FOLDER, 'data')
results_folder = op.join(RESULTS_FOLDER, 'plusmaze')
figure_location = op.join(FIGURE_FOLDER, 'packard')
if not op.exists(figure_location):
    os.makedirs(figure_location)


def load_original_data():
    data = pd.read_csv(op.join(packard_folder, 'plusmaze_ratdata.csv'))
    agg = data.pivot_table(index=['Injection site', 'Test day', 'Treatment', 'Behaviour'], aggfunc=len, margins=True)
    agg['total'] = agg.groupby(['Injection site', 'Test day', 'Treatment']).transform('sum')
    agg['Percentage'] = agg['Run'] / agg['total'] * 100
    agg = agg.drop('All')
    agg = agg.sort_index(ascending=False, level=2)
    agg = agg.reset_index()
    return agg[agg['Behaviour'] == 'Place']


def load_model_data():
    most_recent_runs = {}
    for group in ['control', 'inactivate_HPC', 'inactivate_DLS']:
        rf = op.join(results_folder, group)
        most_recent_runs[group] = get_most_recent_model_run(rf)

    control_results = pd.read_csv(op.join(results_folder, 'control', most_recent_runs['control'], 'summary.csv'))
    hpc_inact_results = pd.read_csv(op.join(results_folder, 'inactivate_HPC', most_recent_runs['inactivate_HPC'], 'summary.csv'))
    dls_inact_results = pd.read_csv(op.join(results_folder, 'inactivate_DLS', most_recent_runs['inactivate_DLS'], 'summary.csv')) # '2019-11-19 12:04:57.620447'

    df = pd.concat([control_results, hpc_inact_results, dls_inact_results])
    df = df.pivot_table(index=['trial', 'group', 'score'], aggfunc=len, margins=True)
    df['total'] = df['agent'].groupby(['trial', 'group']).sum()
    df['Percentage'] = df['agent'] / df['total'] * 100
    df = df.drop('All')
    df = df.reset_index()
    return df[df['score'] == 'place']


def get_most_recent_model_run(directory):
    times = [datetime.strptime(i, '%Y-%m-%d %H:%M:%S.%f') for i in os.listdir(directory)]
    times.sort()
    return str(times[-1])


def performance_barchart():
    ax = sns.catplot(x="Injection site", y="Percentage", hue="Treatment", data=df,
                     kind="bar", col="Test day")
    ax.set_ylabels('% place strategy')
    plt.savefig(op.join(figure_location, 'pm_barchart.pdf'), format='pdf')
    return ax


def performance_pointplot(ax):
    sns.pointplot(ax=ax, data=df[df['Injection site'] == 'Hippocampus'], x='Test day', y='Percentage', hue='Treatment',
                  palette=[col_pal[0], col_pal[3]], markers=["o", "o"])

    cp2 = sns.color_palette('pastel')
    sns.pointplot(ax=ax, data=df[df['Injection site'] == 'Striatum'], x='Test day', y='Percentage', hue='Treatment',
                  palette=[cp2[0], col_pal[2]], markers=["o", "o"])

    leg_handles = ax.get_legend_handles_labels()[0]
    ax.legend(leg_handles, ['Control - saline HPC',
                            'Inactivate HPC - lidocaine',
                            'Control - saline DLS',
                            'Inactivate DLS - lidocaine'], title='Treatment')
    plt.ylabel('% place strategy')
    plt.ylim([0, 100])
    plt.savefig(op.join(figure_location, 'pm_originaldata_pointplot.pdf'), format='pdf')


def performance_pointplot_model(ax, df):
    sns.pointplot(ax=ax, data=df, x='trial', y='Percentage', hue='group',
                  palette=[col_pal[0], col_pal[2], col_pal[3]])
    leg_handles = ax.get_legend_handles_labels()[0]
    ax.legend(leg_handles, ['Control - full model',
                            'Inactivate DLS - only SR',
                            'Inactivate HPC - only MF'], title='Model')
    plt.ylabel('% place strategy')
    plt.ylim([0, 100])
    plt.xlabel('Trial')
    ax.set_xticklabels(['Early', 'Late'])
    plt.savefig(op.join(figure_location, 'pm_model_pointplot.pdf'), format='pdf')


if __name__ == '__main__':
    # plot model results
    df = load_model_data()
    fig = plt.figure(figsize=(5, 4.5))
    ax = fig.add_subplot()
    performance_pointplot_model(ax, df)
    plt.show()

    # Plot original data
    df = load_original_data()
    fig = plt.figure()
    ax = performance_barchart()
    plt.show()

    fig2 = plt.figure(figsize=(5, 4.5))
    ax2 = fig2.add_subplot()
    performance_pointplot(ax2)
    plt.show()

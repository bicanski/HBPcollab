import numpy as np
import pandas as pd
import os

from definitions import RESULTS_FOLDER


def is_common_or_rare(action, out):
    left_outcomes = (5, 6)
    right_outcomes = (7, 8)
    if action == 0 and out in left_outcomes:
        return 'common'
    elif action == 0 and out in right_outcomes:
        return 'rare'
    elif action == 1 and out in left_outcomes:
        return 'rare'
    elif action == 1 and out in right_outcomes:
        return 'common'
    else:
        raise ValueError('The combination of action and outcome does not make sense')


def add_relevant_columns(dataframe):
    dataframe['PreviousAction'] = dataframe.groupby(['Agent'])['Action1'].shift(1)
    dataframe['PreviousStart'] = dataframe.groupby(['Agent'])['StartState'].shift(1)
    dataframe['PreviousReward'] = dataframe.groupby(['Agent'])['Reward'].shift(1)
    dataframe['Stay'] = (dataframe.PreviousAction == dataframe.Action1)
    dataframe['Transition'] = np.vectorize(is_common_or_rare)(dataframe['Action1'], dataframe['Terminus'])
    dataframe['PreviousTransition'] = dataframe.groupby(['Agent'])['Transition'].shift(1)


if __name__ == '__main__':

    data_lesion_hpc = pd.read_csv(os.path.join(RESULTS_FOLDER, 'twostep', 'results_lesion_hpc.csv'))
    data_lesion_dls = pd.read_csv(os.path.join(RESULTS_FOLDER, 'twostep', 'results_lesion_dls.csv'))
    data_control = pd.read_csv(os.path.join(RESULTS_FOLDER, 'twostep', 'results_control.csv'))



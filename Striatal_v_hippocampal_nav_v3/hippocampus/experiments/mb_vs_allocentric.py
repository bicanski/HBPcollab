from tqdm import tqdm
import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
import os
import matplotlib.pyplot as plt

from hippocampus.analysis.daw_analysis import add_relevant_columns
from hippocampus.environments import TwoStepTask
from hippocampus.experiments.twostep import CombinedAgent
from hippocampus.agents import CombinedAgent as SpatialAgent
from hippocampus.environments import HexWaterMaze
from definitions import RESULTS_FOLDER

res_dir = os.path.join(RESULTS_FOLDER, 'mb_spatialmemory')
if not os.path.exists(res_dir):
    os.makedirs(res_dir)


def run_twostep_task(num_agents, params, lesion_hpc=False, **kwargs):
    for a in range(num_agents):
        if lesion_hpc:
            filename = 'twostep_agent{}_lesion'.format(a)
        elif 'inact_hpc' in kwargs:
            filename = 'twostep_partial_lesion_agent{}'.format(a)
        else:
            filename = 'twostep_agent{}'.format(a)

        if os.path.exists(os.path.join(res_dir, filename)):
            tqdm.write('Already done agent {}'.format(a))
            continue

        df = pd.DataFrame({})
        e = TwoStepTask()
        ag = CombinedAgent(e, A_alpha=params['A_alpha'][a], alpha1=params['alpha1'][a],
                           A_beta=params['A_beta'][a], beta1=params['beta1'][a], lesion_hpc=lesion_hpc,
                           inact_hpc=kwargs['inact_hpc'][a])

        for ep in tqdm(range(e.n_trials), leave=False):
            results = ag.one_episode()
            results['Trial'] = ep
            results['Agent'] = 0
            df = df.append(results, ignore_index=True)

        df.to_csv(os.path.join(res_dir, filename))


def determine_platform_seq(platform_states, env):
    indices = np.arange(len(platform_states))
    usage = np.zeros(len(platform_states))

    plat_seq = [np.random.choice(platform_states)]
    for sess in range(1, 11):
        distances = np.array([env.grid.distance(plat_seq[sess - 1], s) for s in platform_states])
        candidates = indices[np.logical_and(usage < 2, distances > env.grid.radius)]
        platform_idx = np.random.choice(candidates)
        plat_seq.append(platform_states[platform_idx])
        usage[platform_idx] += 1.

    return plat_seq


def run_watermaze(num_agents, params, lesion_hpc=False, **kwargs):
    g = HexWaterMaze(6)

    for i_a in tqdm(range(num_agents), desc='Agent'):

        if lesion_hpc:
            filename = 'spatial_agent{}_lesion'.format(i_a)
        elif 'inact_hpc' in kwargs:
            filename = 'spatial_partial_lesion_agent{}'.format(i_a)
        else:
            filename = 'spatial_agent{}'.format(i_a)

        if os.path.exists(os.path.join(res_dir, filename)):
            tqdm.write('Already done')
            continue

        #possible_platform_states = np.array([192, 185, 181, 174, 216, 210, 203, 197])  # for the r = 10 case
        possible_platform_states = np.array([48, 45, 42, 39, 60, 57, 54, 51])

        platform_sequence = determine_platform_seq(possible_platform_states, g)

        # intialise agent
        agent = SpatialAgent(g, init_sr='rw', A_alpha=params['A_alpha'][i_a], alpha1=params['alpha1'][i_a],
                           A_beta=params['A_beta'][i_a], beta1=params['beta1'][i_a], lesion_hpc=lesion_hpc,
                             inact_hpc=kwargs['inact_hpc'][i_a])
        agent_results = []

        total_trial_count = 0

        for ses in tqdm(range(11), desc='Session', leave=False):
            for trial in tqdm(range(4), leave=False, desc='Trial'):
                # every first trial of a session, change the platform location
                if trial == 0:
                    g.set_platform_state(platform_sequence[ses])

                res = agent.one_episode(random_policy=False)
                res['trial'] = trial
                res['escape time'] = res.time.max()
                res['session'] = ses
                res['total trial'] = total_trial_count
                agent_results.append(res)
                total_trial_count += 1

        agent_df = pd.concat(agent_results)
        agent_df['total time'] = np.arange(len(agent_df))

        agent_df.to_csv(os.path.join(res_dir, filename))


def get_states_around_last_platform(env, rec_depth=2):
    platform_state = env.previous_platform_state
    states = [platform_state]
    for i in range(rec_depth):
        added_states = []
        for s in states:
            neighbours = np.flatnonzero(env.adjacency_graph[s])
            for n in neighbours:
                if n not in states and n not in added_states:
                    added_states.append(n)
        states += added_states
    return states


# model based model free analysis
def get_model_weights(data):
    add_relevant_columns(data)
    data['Stay'] = data['Stay'].astype('int')
    data = data[['Stay', 'PreviousReward', 'PreviousTransition']]
    mod = smf.logit(formula='Stay ~ PreviousTransition * PreviousReward', data=data)
    res = mod.fit()
    model_based_weight = -res.params['PreviousTransition[T.rare]:PreviousReward']
    model_free_weight = res.params['PreviousReward']
    return model_based_weight, model_free_weight


if __name__ == "__main__":
    # Sample some parameters

    n_agents = 20
    np.random.seed(10)

    # loop over different parameter values for the transitions from MF to SR and vice versa
    A_alpha = np.linspace(.5, 5, n_agents)
    alpha1 = np.linspace(.01, 2, n_agents)
    A_beta = np.linspace(2, .5, n_agents)
    beta1 = np.linspace(.3, .1, n_agents)

    parameters = pd.DataFrame({})
    parameters['A_alpha'] = A_alpha
    parameters['alpha1'] = alpha1
    parameters['A_beta'] = A_beta
    parameters['beta1'] = beta1

    parameters.to_csv(os.path.join(res_dir, 'params.csv'))

    # healthy control
    tqdm.write('\n RUNNING TWO-STEP TASK \n')
    run_twostep_task(n_agents, parameters)
    tqdm.write('\n RUNNING WATER MAZE \n')
    run_watermaze(n_agents, parameters)

    # hippocampal lesion
    tqdm.write('\n RUNNING TWO-STEP TASK (HPC LESION) \n')
    run_twostep_task(n_agents, parameters, lesion_hpc=True)
    tqdm.write('\n RUNNING WATER MAZE (HPC LESION) \n')
    run_watermaze(n_agents, parameters, lesion_hpc=True)

    # hippocampal partial lesion
    min_inact_prop = .65
    np.random.seed(5)
    inact_hpc = np.random.uniform(min_inact_prop, 1., n_agents)

    tqdm.write('\n RUNNING TWO-STEP TASK (HPC PARTIAL LESION) \n')
    run_twostep_task(n_agents, parameters, lesion_hpc=False, inact_hpc=inact_hpc)
    tqdm.write('\n RUNNING WATER MAZE (HPC PARTIAL LESION) \n')
    run_watermaze(n_agents, parameters, lesion_hpc=False, inact_hpc=inact_hpc)


    tqdm.write('done')

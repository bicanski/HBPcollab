import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import os
from datetime import datetime
from multiprocessing import Pool

from hippocampus import utils
from hippocampus.environments import PlusMaze
from hippocampus.agents import SRTD, LandmarkLearningAgent, CombinedAgent
from definitions import ROOT_FOLDER


groups = {0: 'control',
          1: 'inactivate_HPC',
          2: 'inactivate_DLS'}

group = groups[2]

# save location
results_folder = os.path.join(ROOT_FOLDER, 'results', 'plusmaze', group, str(datetime.now()))
figure_folder = os.path.join(results_folder, 'figures')
if not os.path.exists(results_folder):
    os.makedirs(results_folder)
    os.makedirs(figure_folder)


# 4 trials per day. 14 total training days. 2 test days.
#  50 rats
# The entrance to the south maze arm
# (i.e., the arm containing the start box used during DISCUSSION training) was blocked by a clear Plexiglas shield on
# the probe trials.

if group == 'inactivate_HPC':
    inactivate_HPC = .3
else:
    inactivate_HPC = False
if group =='inactivate_DLS':
    inactivate_DLS = .6
else:
    inactivate_DLS = False

n_agents = 40
n_trials = 200  # 4 * 14 + 2

pm = PlusMaze()

agent_ids = np.arange(n_agents)


def run_plusmaze_task(n_agent):
    behavioural_scores = pd.DataFrame({})

    np.random.seed(n_agent)

    agent = CombinedAgent(env=pm, inact_hpc=inactivate_HPC, inact_dls=inactivate_DLS, learning_rate=.07, inv_temp=5,
                          eta=.03)

    agent_results = []

    for trial in tqdm(range(n_trials), leave=False):
        if trial == 20 or trial == n_trials - 1:
            agent.env.toggle_probe_trial()
        else:
            agent.env.toggle_training_trial()

        res = agent.one_episode(random_policy=False)
        res['trial'] = trial
        res['escape time'] = res.time.max()
        res['goal location'] = agent.env.get_goal_state()
        res['total reward'] = res['reward'].sum()
        last_state = res['state'].iloc[-2]
        res['last state'] = last_state
        res['trial type'] = agent.env.trial_type
        if agent.env.trial_type == 'probe':
            if last_state == agent.env.rewarded_terminal:
                res['score'] = 'place'
            elif last_state == 5:
                res['score'] = 'response'

            behavioural_scores = behavioural_scores.append({'agent': n_agent,
                                                            'trial': trial,
                                                            'score': res['score'].iloc[0],
                                                            'group': group}, ignore_index=True)
        else:
            if last_state == agent.env.goal_state:
                res['score'] = 'correct'
            elif last_state == 5:
                res['score'] = 'incorrect'
        agent_results.append(res)

    df = pd.concat(agent_results, sort=True)
    df['agent'] = n_agent
    df.to_csv(os.path.join(results_folder, 'agent{}.csv'.format(n_agent)))
    return behavioural_scores

#run_plusmaze_task(0)
p = Pool(os.cpu_count() - 1)
results = p.imap(run_plusmaze_task, agent_ids)
behavioural_scores = pd.concat(results)

behavioural_scores.to_csv(os.path.join(results_folder, 'summary.csv'))

agg = behavioural_scores.pivot_table(index=['trial', 'score'], aggfunc=len, margins=True)


plt.figure()
ax = sns.barplot(x='trial', y='agent', hue='score', data=agg.reset_index())
plt.show()
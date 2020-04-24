import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from hippocampus.agents import LandmarkLearningAgent, CombinedAgent
from hippocampus.environments import HexWaterMaze


if __name__ == '__main__':
    from tqdm import tqdm
    from hippocampus.plotting import tsplot_boot
    g = HexWaterMaze(5)
    g.set_platform_state(30)

    all_ets = []
    for ag in tqdm(range(5)):

        agent = LandmarkLearningAgent(g)

        agent_results = []
        agent_ets = []
        for ep in range(60):
            res = agent.one_episode()
            res['trial'] = ep
            res['escape time'] = res.time.max()
            agent_results.append(res)
            agent_ets.append(res.time.max())
        all_ets.append(agent_ets)

    fig, ax = plt.subplots()
    tsplot_boot(ax, np.array(all_ets))
    plt.show()

    df = pd.concat(agent_results)

    df.plot(x='trial', y=['escape time'], legend=False)
    plt.show()

    fig, ax = plt.subplots()
    plt.plot(np.array(df.reliability), alpha=.5)
    plt.plot(np.array(df.RPE), alpha=.5)
    plt.plot(np.array(df.reward), alpha=.5, color='k')

    plt.legend(['Reliability', 'RPE', 'Reward'])
    plt.xlabel('Time')

    plt.show()
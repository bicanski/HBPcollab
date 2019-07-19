import numpy as np

from Experiments.Blocking.environment import Environment
import matplotlib.pyplot as plt

envi = Environment()

from Experiments.Blocking.Blocking import Agent, set_cues

ag = Agent()

set_cues('A', ag)

rm_A = ag.hippocampus.get_ratemap()

set_cues('both', ag)

rm_both = ag.hippocampus.get_ratemap()

set_cues('B', ag)

rm_B = ag.hippocampus.get_ratemap()




fig, ax = plt.subplots(8, 5)

for i in range(rm_A.shape[0]):
    idx = np.unravel_index(i, (8, 5))
    plt.sca(ax[idx])
    ax[idx].set_xticks([])
    ax[idx].set_yticks([])
    plt.imshow(rm_A[i])

plt.savefig('A.png')



fig, ax = plt.subplots(8, 5)

for i in range(rm_A.shape[0]):
    idx = np.unravel_index(i, (8, 5))
    plt.sca(ax[idx])
    ax[idx].set_xticks([])
    ax[idx].set_yticks([])
    plt.imshow(rm_B[i])

plt.savefig('B.png')




fig, ax = plt.subplots(8, 5)

for i in range(rm_A.shape[0]):
    idx = np.unravel_index(i, (8, 5))
    plt.sca(ax[idx])
    ax[idx].set_xticks([])
    ax[idx].set_yticks([])
    plt.imshow(rm_both[i])

plt.savefig('both.png')
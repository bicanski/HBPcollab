import numpy as np
from hebbian_hippocampus import Hippocampus, CognitiveHippocampus, TDHippocampus
from striatum_landmarkcells import Striatum, CognitiveStriatum, TDStriatum
from environments import WaterMazeEnv, PlusMaze
from cognitive_tasks import DeterministicTask, StochasticTask
import matplotlib.pyplot as plt
from plotting import tsplot_boot
from tqdm import tqdm
import pandas as pd
import os


class TDAgent(object):

    def __init__(self, env=WaterMazeEnv(), epsilon=.1, zeta=1.2, lesion_striatum=False, lesion_hippocampus=False):
        """Initialise the agent with a hippocampus and striatum. If both are lesioned, behaviour will be random.

        :param (float) epsilon: Randomness parameter.
        :param (bool) lesion_striatum: Inactivates the striatum model.
        :param (bool) lesion_hippocampus: Inactivates the hippocampus model.
        """
        self.epsilon = epsilon
        self.zeta = zeta
        if lesion_hippocampus and lesion_striatum:
            self.epsilon = 1  # Output random behaviour if no striatum and no hippocampus are present.
        self.striatum_lesion = lesion_striatum
        self.hippocampus_lesion = lesion_hippocampus
        self.env = env
        self.hippocampus = TDHippocampus(self.env)
        self.striatum = TDStriatum(self.env)
        self.reached_goal = False

    def train_one_episode(self):
        self.env.reset()
        self.reached_goal = False
        reached_terminus = False
        self.striatum.update()
        self.hippocampus.update()

        t = 0
        reward = 0

        locs = [[self.env.curr_x, self.env.curr_y]]
        choices = []

        while not self.reached_goal and not reached_terminus and t < self.env.timeout / self.env.time_bin:

            action, expert = self.choose_action()
            self.reached_goal = self.env.act(action)
            if self.reached_goal:
                reward = 1

            if isinstance(self.env, PlusMaze):
                reached_terminus = self.env.agent_at_terminus()

            self.hippocampus.update()
            self.striatum.update()

            t += 1

            locs.append([self.env.curr_x, self.env.curr_y])
            choices.append(expert)

        return t, reward, np.array(locs), choices

    def choose_action(self):
        """Choose action from both hippocampus and striatum and compare their value.
        """
        if np.random.rand() <= self.epsilon:
            action = np.random.choice(self.env.actions)
            return action, 'random'

        hc_action, hc_value = self.hippocampus.choose_action()  # output direction in degrees.
        str_action, str_value = self.striatum.choose_action()

        if self.hippocampus_lesion:
            return str_action, 'striatum'
        elif self.striatum_lesion:
            return hc_action, 'hippocampus'

        if hc_value == str_value:
            choice = np.random.choice(['striatum', 'hippocampus'])
            if choice == 'hippocampus':
                return hc_action, 'hippocampus'
            else:
                return str_action, 'striatum'
        elif hc_value > self.zeta * str_value:
            return hc_action, 'hippocampus'
        else:
            return str_action, 'striatum'


class Agent(object):
    """Agent containing a hippocampus and striatum for decision making. The value output of both brain areas are
    compared to make the final decision.
    """
    def __init__(self, env=WaterMazeEnv(), epsilon=.1, lesion_striatum=False, lesion_hippocampus=False):
        """Initialise the agent with a hippocampus and striatum. If both are lesioned, behaviour will be random.

        :param (float) epsilon: Randomness parameter.
        :param (bool) lesion_striatum: Inactivates the striatum model.
        :param (bool) lesion_hippocampus: Inactivates the hippocampus model.
        """
        self.epsilon = epsilon
        if lesion_hippocampus and lesion_striatum:
            self.epsilon = 1  # Output random behaviour if no striatum and no hippocampus are present.
        self.striatum_lesion = lesion_striatum
        self.hippocampus_lesion = lesion_hippocampus
        self.env = env
        self.hippocampus = Hippocampus(self.env)
        self.striatum = Striatum(self.env)
        self.reached_goal = False

    def train_one_episode(self):
        self.env.reset()
        self.reached_goal = False
        reached_terminus = False
        self.striatum.update()
        self.hippocampus.update()

        t = 0
        reward = 0

        locs = [[self.env.curr_x, self.env.curr_y]]
        choices = []

        while not self.reached_goal and not reached_terminus and t < self.env.timeout / self.env.time_bin:

            action, expert = self.choose_action()
            self.reached_goal = self.env.act(action)

            if isinstance(self.env, PlusMaze):
                reached_terminus = self.env.agent_at_terminus()

            self.hippocampus.update()
            if self.reached_goal:
                reward += self.env.reward
                self.hippocampus.goal_reached()

            self.striatum.update()

            t += 1

            locs.append([self.env.curr_x, self.env.curr_y])
            choices.append(expert)

        return t, reward, np.array(locs), choices

    def choose_action(self):
        """Choose action from both hippocampus and striatum and compare their value.
        """
        if np.random.rand() <= self.epsilon:
            action = np.random.choice(self.env.actions)
            return action, 'random'

        hc_action, hc_value = self.hippocampus.choose_action()  # output direction in degrees.
        str_action, str_value = self.striatum.choose_action()

        if self.hippocampus_lesion:
            return str_action, 'striatum'
        elif self.striatum_lesion:
            return hc_action, 'hippocampus'

        if hc_value == str_value:
            choice = np.random.choice(['striatum', 'hippocampus'])
            if choice == 'hippocampus':
                return hc_action, 'hippocampus'
            else:
                return str_action, 'striatum'
        elif hc_value > str_value:
            return hc_action, 'hippocampus'
        else:
            return str_action, 'striatum'


class NonSpatialAgent(object):
    def __init__(self, env=DeterministicTask(), epsilon=.2, lesion_striatum=False, lesion_hippocampus=False):
        self.env = env
        self.epsilon = epsilon
        self.hippocampus = CognitiveHippocampus(self.env, learning_rate=.9)
        self.striatum = CognitiveStriatum(self.env, learning_rate=.8, lamb=0.8)

        self.hippocampus_lesion = lesion_hippocampus
        self.striatum_lesion = lesion_striatum

    def train_one_episode(self):
        t = 0
        reward = 0
        self.env.reset()
        reached_terminus = False
        self.hippocampus.update()
        self.striatum.eligibility_trace = np.zeros(self.striatum.eligibility_trace.shape)
        self.striatum.LC_activations = np.zeros(self.striatum.LC_activations.shape)
        self.striatum.update(reward)
        data_to_write = [self.env.start_state]

        while not reached_terminus:
            self.env.reward_probs = self.env.reward_traces[:, 0]

            action, expert = self.choose_action()
            next_state, reward = self.env.act(action)
            self.hippocampus.update()
            self.striatum.update(reward)
            reached_terminus = self.env.is_terminal(self.env.curr_state)
            if reached_terminus and reward > 0:
                self.hippocampus.goal_cell_rate = 1
                self.hippocampus.update_weights()
            elif reached_terminus and reward == 0:
                self.hippocampus.goal_cell_rate = 0
                self.hippocampus.update_weights()

            t += 1
            data_to_write.append(action)

        data_to_write.append(self.env.curr_state)
        data_to_write.append(reward)
        return data_to_write

    def choose_action(self):
        if np.random.rand() <= self.epsilon:
            action = np.random.choice(self.env.get_possible_actions(self.env.curr_state))
            return action, 'random'

        hc_action, hc_value = self.hippocampus.choose_action()
        str_action, str_value = self.striatum.choose_action()

        if self.hippocampus_lesion:
            return str_action, 'striatum'
        elif self.striatum_lesion:
            return hc_action, 'hippocampus'

        if hc_value == str_value:
            choice = np.random.choice(['striatum', 'hippocampus'])
            if choice == 'hippocampus':
                return hc_action, 'hippocampus'
            else:
                return str_action, 'striatum'
        elif hc_value > str_value:
            return hc_action, 'hippocampus'
        else:
            return str_action, 'striatum'


if __name__ == '__main__':
    n_agents = 23
    n_episodes = 272
    all_rewards = np.zeros((n_agents, n_episodes))
    all_escape_times = np.zeros((n_agents, n_episodes))

    df = pd.DataFrame(columns=['Agent_nr', 'Trial', 'StartState', 'Action1', 'Action2', 'Terminus', 'Reward'])
    for agent in tqdm(range(n_agents)):
        #a = NonSpatialAgent(env=DeterministicTask(), lesion_hippocampus=True, lesion_striatum=False, epsilon=0.2)
        a = Agent(env=WaterMazeEnv(), lesion_hippocampus=True)
        for ep in range(n_episodes):
            data = a.train_one_episode()
            all_escape_times[agent, ep] = data[0]
            #all_rewards[agent, ep] = data[-1]
            #df.loc[len(df)] = [agent, ep] + data

    df.to_csv(os.path.join(a.env.output_folder, 'BehaviourDriftingRewardsStriatum23July.csv'))

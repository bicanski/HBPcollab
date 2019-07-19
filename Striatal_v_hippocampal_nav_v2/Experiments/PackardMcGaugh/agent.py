import numpy as np
from Experiments.PackardMcGaugh.hippocampus import TDHippocampus
from Experiments.PackardMcGaugh.striatum import TDStriatum
from environments import WaterMazeEnv, PlusMaze


class TDAgent(object):

    def __init__(self, env=PlusMaze(), epsilon=.1, zeta=1.2, lesion_striatum=False, lesion_hippocampus=False):
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

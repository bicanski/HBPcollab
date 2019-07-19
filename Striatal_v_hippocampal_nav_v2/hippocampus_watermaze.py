import math

import numpy as np
from matplotlib import pyplot as plt, cm
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

import utils
from environments import WaterMazeEnv
from watermaze import WaterMazeAgent


class HippocampalAgent(WaterMazeAgent):
    """This class uses the allocentric algorithm to perform the Morris Water Maze task, using place cells.
    """

    def __init__(self, n_trials=30, env=WaterMazeEnv()):

        WaterMazeAgent.__init__(self, n_trials=n_trials, env=env)
        self.env.trial = 0

        self.field_centres = self.create_place_cells()
        self.field_width = .09
        self.max_response = utils.gauss2d([0, 0], self.field_width ** 2, [0, 0])
        self.place_cell_responses = np.zeros(self.field_centres[0].shape)
        self.previous_place_cell_responses = np.zeros(self.field_centres[0].shape)
        self.update_place_cell_response()

        self.actions = {idx: direction for (idx, direction) in zip(range(12), range(0, 360, 30))}
        self.actions[0] = 360

        self.current_action = np.random.choice(len(self.actions))
        self.previous_action = None

        # initialise critic:
        self.critic_weights = np.zeros(self.place_cell_responses.shape)
        self.max_sum_weights = 1.
        self.critic_activation = np.dot(self.critic_weights, self.place_cell_responses)
        self.previous_critic_activation = None

        # initialise actor:
        self.action_weights = np.zeros((len(self.actions), self.place_cell_responses.shape[0]))
        self.action_values = np.dot(self.action_weights, self.place_cell_responses)
        self.policy = self.softmax(self.action_values)

        self.policies = []
        self.policies.append(self.evaluate_policy_at_field_centres())

    def run_simulation(self):
        """Run the WaterMaze task for a set number of trials.
        """
        for _ in tqdm(range(self.n_trials)):
            self.trial_reset()
            reached_platform = False
            while not reached_platform and self.env.t < 120/self.env.time_bin:
                reached_platform = self.make_step()
            self.policies.append(self.evaluate_policy_at_field_centres())

    def make_step(self):
        """Update the agent for one time step: compute new policy, update the position accordingly, incur any potential
        reward, update the critic and actor accordingly.
        """
        self.env.t += 1
        self.env.total_time += 1
        self.determine_policy()
        self.update_position()
        self.reward = self.check_reward()
        self.log_positions()
        self.update_place_cell_response()
        self.update_critic()
        self.update_actor()
        self.total_reward += self.reward
        self.reward = 0
        return self.on_platform()

    def update_place_cell_response(self):
        """Compute the firing rate of a each place cell with Gaussian tuning curves, for the current position of the
        agent
        """
        self.previous_place_cell_responses = self.place_cell_responses
        self.place_cell_responses = utils.gauss2d(self.current_position, var=self.field_width ** 2,
                                                  centre=self.field_centres) / self.max_response

    def compute_place_cell_response(self, position):
        """Compute the responses of all place cells.

        :param position: Current position (x,y) of the agent.
        :return:
        """
        return utils.gauss2d(position, var=self.field_width ** 2, centre=self.field_centres) / self.max_response

    def create_place_cells(self):
        """Create a grid of evenly spaced place field centres within the water maze.
        """
        field_centres_x, field_centres_y = np.meshgrid(np.linspace(-1, 1, 25), np.linspace(-1, 1, 25))
        field_centres = np.array([field_centres_x.flatten(), field_centres_y.flatten()])
        field_centres = field_centres[:, self.within_maze(field_centres)]
        return field_centres

    def update_position(self):
        self.previous_action = self.current_action
        policy_sampled_action = np.random.choice(len(self.actions), p=self.policy)
        self.current_action = self.apply_momentum(policy_sampled_action)
        position_update = self.get_position_update(self.current_action)
        self.check_wall_collision(position_update)
        self.current_position += position_update

    def trial_reset(self):
        """Reset the agent to its starting position.
        """
        self.env.t = 0
        self.reward = 0
        self.current_position = self.initial_position
        self.delta = 0
        self.env.trial += 1
        self.previous_critic_activation = 0
        self.critic_activation = 0
        self.previous_place_cell_responses = np.zeros(self.field_centres[0].shape)

    def _compute_prediction_error(self):
        """Compute the reward prediction error as the difference between the critic's current and previous activation,
        plus any received rewards.
        """
        self.delta = self.reward + self.gamma * self.critic_activation - \
                     self.previous_critic_activation

    def compute_prediction_error(self):
        """Compute the reward prediction error as the difference between the critic's current and previous activation,
        plus any received rewards.
        """
        if self.reward == 1:
            self.delta = self.reward - self.previous_critic_activation
        else:
            self.delta = self.gamma * self.critic_activation - self.previous_critic_activation

    def update_critic(self):
        """Update the 'critic' of the learner using a TD update rule
        """
        self.previous_critic_activation = self.critic_activation
        self.critic_activation = np.dot(self.critic_weights, self.place_cell_responses)
        self.compute_prediction_error()
        self.critic_weights += self.delta * self.previous_place_cell_responses * self.learning_rate

    def update_actor(self):
        """Update the weight matrix from the place cells to the actor cells using the TD learning rule.
        """
        self.action_weights[self.previous_action] += self.delta * \
            self.previous_place_cell_responses * \
            self.learning_rate

    def determine_policy(self):
        """Use the action weight matrix to compute the value of each possible action. Then apply softmax to get policy.
        """
        self.action_values = np.dot(self.action_weights, self.place_cell_responses)
        self.policy = self.softmax(self.action_values)

    def value_function(self, position):
        """Compute the value function at a given position.

        :param position: (x,y) coordinate.
        :return:
        """
        place_cell_responses = self.compute_place_cell_response(position)
        return np.dot(self.critic_weights, place_cell_responses)

    def evaluate_policy_at_field_centres(self):
        """At each place field centre, evaluate what the agent's current policy would be given its current weights.
        """
        policies = np.empty((len(self.place_cell_responses), len(self.policy)))
        for idx, field in enumerate(self.field_centres.T):
            place_cell_responses = self.compute_place_cell_response(field)
            action_values = np.dot(self.action_weights, place_cell_responses)
            policies[idx] = self.softmax(action_values)
        return policies

    def plot_quiver(self, policies=None):
        """Draw a quiver plot of the current preferred directions at each place field centre.

        :param policies: Array of current policies to be computed with self.evaluate_policy_at_field_centres()
        """
        if policies is None:
            policies = self.evaluate_policy_at_field_centres()
        chosen_actions = np.argmax(policies, axis=1)
        probability = np.max(policies, axis=1)
        preferred_vector = np.empty(self.field_centres.shape)

        for i, action in enumerate(chosen_actions):
            preferred_vector[:, i] = self.get_position_update(action)  #self.actions[action]

        preferred_vector = np.multiply(probability, preferred_vector)  # scale vector to its probability

        fig, ax = plt.subplots()
        self.draw_maze_and_platform(ax)

        x = self.field_centres[0]
        y = self.field_centres[1]
        u = preferred_vector[0]
        v = preferred_vector[1]
        plt.axis('equal')
        plt.quiver(x, y, u, v)
        ax.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off')

    def evaluate_value_function_everywhere(self):
        xs = np.linspace(-1, 1, 50)
        ys = np.linspace(-1, 1, 50)

        z = np.zeros((xs.shape[0], ys.shape[0]))

        for i in range(xs.shape[0]):
            for j in range(ys.shape[0]):
                z[i, j] = self.value_function([xs[i], ys[j]])

        x, y = np.meshgrid(xs, ys)
        return x, y, z

    def plot_value_function(self):
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        x, y, z = self.evaluate_value_function_everywhere()
        ax.plot_surface(x, y, z, linewidth=0, antialiased=False, cmap=cm.coolwarm)

    def plot_place_field(self, centre):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        covariance_matrix = np.eye(2) * self.field_width ** 2
        z = np.exp(-((self.xs - centre[0]) ** 2 + (self.ys - centre[1]) ** 2) / (2 * self.field_width ** 2)) / \
            np.sqrt((2 * np.pi) ** 2 * np.linalg.det(covariance_matrix))
        ax.plot_surface(self.xs, self.ys, z, linewidth=0, antialiased=False)

    def softmax(self, x):
        """Compute the softmax probability distribution for a vector x.
        """
        return np.exp(self.beta * x) / sum(np.exp(self.beta * x))


if __name__ == '__main__':
    h = HippocampalAgent()
    h.run_simulation()
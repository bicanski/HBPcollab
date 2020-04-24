"""This module contains methods to find value functions and policies for any given environment using
dynamic programming methods."""

import numpy as np

from hippocampus.utils import all_argmax


def generate_random_policy(env):
    """Generate a random policy assigning equal probability to all possible actions in each state.

    :param env: Environment object to be evaluated.
    :return: Nested list[state][action] giving the probabilities of choosing [action] when in [state].
    """
    random_policy = []
    for state in env.state_indices:
        possible_actions = env.get_possible_actions(state)
        if not possible_actions:
            random_policy.append([[]])
            continue
        rand_pol = [1 / len(possible_actions)] * len(possible_actions)
        random_policy.append(rand_pol)
    return random_policy


def policy_evaluation(env, policy, theta=1e-4, gamma=.9):
    """Implementation of the iterative policy evaluation algorithm from dynamic programming. Finds state values
    associated to a given policy

    :param env: Environment object to be evaluated.
    :param policy: Policy to evaluate (nested list[state][action], such as the output of generate_random_policy)
    :param theta: Cutoff criterion for convergence of the optimal policy.
    :param gamma: Exponential discount factor for future rewards.
    :return:
    """
    values = np.zeros(env.nr_states)
    while True:
        delta = 0
        for origin in env.state_indices:
            old_value = values[origin]
            weighted_action_values = []
            for action in env.get_possible_actions(origin):
                transition_prob = env.transition_probabilities[origin, action]
                action_value = np.dot(transition_prob, env.reward_func[origin, action] + gamma * values)
                weighted_action_values.append(policy[origin][action] * action_value)
            values[origin] = sum(weighted_action_values)
            delta = max(delta, abs(old_value - values[origin]))
        if delta < theta:
            break
    return values


def value_iteration(env, theta=1e-4, gamma=.9):
    """Implement the  value iteration algorithm from dynamic programming to compute the optimal policy and corresponding
    state-value function for a given environment.

    :param env: Environment object to be evaluated.
    :param theta: Cutoff criterion for convergence of the optimal policy.
    :param gamma: Exponential discount factor for future rewards.
    :return: List containing the optimal policy, and array containing values for each state.
    """
    optimal_values = find_optimal_values(env, gamma=gamma, theta=theta)
    optimal_policy = optimal_policy_from_value(env, optimal_values, gamma)
    return optimal_policy, optimal_values


def find_optimal_values(env, theta=1e-4, gamma=.9):
    """Find optimal state values in an environment through value iteration.

    :param env: Environment object to be evaluated.
    :param theta: Cutoff criterion for convergence of the optimal policy.
    :param gamma: Exponential discount factor for future rewards.
    :return: Array containing values for each state under the optimal policy.
    """
    optimal_values = np.zeros(env.nr_states)
    while True:
        delta = 0
        for origin in env.state_indices:
            if env.is_terminal(origin):
                optimal_values[origin] = 0
                continue
            old_value = optimal_values[origin]
            action_values = np.zeros(env.nr_actions)
            for action in env.get_possible_actions(origin):
                transition_prob = env.transition_probabilities[origin, action]
                action_values[action] = np.dot(transition_prob, env.reward_func[origin, action] + gamma * optimal_values)
            optimal_values[origin] = max(action_values)
            delta = max(delta, abs(old_value - optimal_values[origin]))
        if delta < theta:
            break
    return optimal_values


def optimal_policy_from_value(env, state_values, gamma=.9):
    """Compute the greedy policy from a given set of state values.
    
    :param env: Environment object to be evaluated.
    :param state_values: Value of all states under the current policy
    :param gamma: Exponential discount factor for future rewards.
    :return: 
    """
    optimal_policy = []
    for state in env.state_indices:
        if len(env.get_possible_actions(state)) == 0:
            optimal_policy.append([])
            continue
        policy = [0] * len(env.get_possible_actions(state))
        action_values = []
        for action in env.get_possible_actions(state):
            transition_prob = env.transition_probabilities[state, action]
            action_values.append(np.dot(transition_prob, env.reward_func[state, action] + gamma * state_values))
        optimal_actions = all_argmax(action_values)

        for idx in optimal_actions:
            policy[idx] = 1 / optimal_actions.size
        optimal_policy.append(policy)
    return optimal_policy


if __name__ == "__main__":
    from hippocampus.environments import LinearTrack
    environ = LinearTrack()
    policy, values = value_iteration(environ)

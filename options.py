'''Class and utility functions related to options'''

from typing import Tuple
from collections import defaultdict
import numpy as np
from qlearning import Qlearning

class Option(object):
    '''Class to represent an option'''
    def __init__(self, initiation_set, goal_state) -> None:
        self.initiation_set = initiation_set
        # instead of a termination condition, we just use a goal state to keep things simple
        self.goal_state = goal_state
        # the q table represents the policy of the option
        self.q_table = None

    def learn_option_policy(self, env, episodes, lr, gamma, eps) -> None:
        '''Learn a policy for the option using Q learning'''
        agent = Qlearning(env)
        agent.train(episodes, lr, gamma, eps)
        self.q_table = agent.q_table

    def execute_option(self, env, state, gamma) -> Tuple[int, float, bool, int, Tuple[int, int]]:
        '''Execute this option using a greedy policy based on the learned q values'''
        total_reward = 0
        steps = 0
        actions_states = []
        while True:
            action = np.argmax(self.q_table[state])
            next_state, reward, done = env.step(action)
            total_reward += reward * (gamma ** steps)
            steps += 1
            actions_states.append((action, next_state))
            if done or next_state == self.goal_state:
                break
            state = next_state
        return next_state, total_reward, done, steps, actions_states

    def __str__(self) -> str:
        return "Init: " + str(self.initiation_set) + " Goal: " + str(self.goal_state)

def _do_interpolation(init_set, env) -> Tuple[int]:
    '''Interpolate the initiation set for the option. Mentioned in the paper'''
    # this function is not tested thoroughly, included for completeness
    expanded_set = []
    for s in init_set:
        expanded_set.extend(env.neighbors(s))
    return list(set(expanded_set))

def generate_options(trajectories, env, NUM_OPTIONS, interpolate=False) -> Tuple[Option]:
    '''Generates options for the given environment using the provided trajectories'''
    state_visitation_counts = defaultdict(int)
    for seq in trajectories:
        for s in seq[:-1]: # the last state is the goal state so skip it
            state_visitation_counts[s] += 1

    selected_goal_states = []
    options = []
    while len(options) < NUM_OPTIONS:
        target = None
        # Select the state present in most trajectories
        for s in sorted(state_visitation_counts, key=state_visitation_counts.get, reverse=True):
            if s not in selected_goal_states:
                selected_goal_states.append(s)
                break
        target = s
        n_s = state_visitation_counts[s]

        # Count the number of other states occur with the most frequently visited state
        common_visit_counts = defaultdict(int)
        for seq in trajectories:
            if target in seq:
                for s in seq:
                    if s != target:
                        common_visit_counts[s] += 1
        avgcounts = np.mean([v for v in common_visit_counts.values()])

        # select states which occur with the most frequeent states more than average
        init_set = [s for s in common_visit_counts.keys() if common_visit_counts[s] > avgcounts]
        # if no state has visitation counts > average it means all have the same value
        if len(init_set) == 0:
            init_set = [s for s in common_visit_counts.keys()]
        if interpolate:
            init_set = _do_interpolation(init_set, env)

        o = Option(initiation_set=init_set, goal_state=target)
        options.append(o)
    return options

def choose_action(q_table, options, state, eps) -> int:
    '''Function to choose an epsilon greedy action among candidates that also include options'''
    valid_actions = [0, 1, 2, 3]
    # The primitive actions are always available. An option is available if
    # the current state is in the initiation set for the option
    for i,o in enumerate(options):
        init_set = o.initiation_set
        if state in init_set:
            valid_actions.append(i + 4)

    q_values = np.array([q for i,q in enumerate(q_table[state]) if i in valid_actions])
    if np.random.random() < eps:
        return np.random.choice(valid_actions)
    return np.random.choice(np.where(q_values == q_values.max())[0]) # randomly break ties

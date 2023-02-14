'''Basic Qlearning algorithm'''

from collections import defaultdict
import itertools
from typing import Tuple
import numpy as np

class Qlearning(object):
    '''Class to implement basic qlearning algorithm'''
    def __init__(self, env) -> None:
        self.env = env

        self.n_actions = len(self.env.actions())
        self.n_states = len(self.env.states())
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions))

    def epsilon_greedy_action(self, q_values, eps) -> int:
        '''Choose an episilong greedy action based on the q values'''
        if np.random.random() < eps:
            return np.random.choice(self.env.actions())
        # return np.argmax(q_values)
        return np.random.choice(np.where(q_values == q_values.max())[0]) # randomly break ties

    def greedy_rollout(self, env) -> Tuple[int]:
        '''Execute the current policy in a greedy fashion and generate a rollout'''
        actions = []
        states = []
        state = env.reset()
        while True:
            action = np.argmax(self.q_table[state])
            next_state, reward, done = env.step(action)
            states.append(next_state)
            actions.append(action)
            if done:
                return states, actions
            state = next_state

    def train(self, episodes=100, lr=0.2, gamma=0.95, eps=0.2) -> Tuple[int]:
        '''Run q learning'''

        episode_lengths = []

        for _ in range(episodes):
            state = self.env.reset()

            for i in itertools.count():

                # Take a step
                action = self.epsilon_greedy_action(self.q_table[state], eps)
                next_state, reward, done = self.env.step(action)

                # TD Update
                next_q_values = self.q_table[next_state]
                best_next_action = np.argmax(next_q_values)
                td_target = reward + gamma * self.q_table[next_state][best_next_action]
                td_delta = td_target - self.q_table[state][action]
                self.q_table[state][action] += lr * td_delta

                if done:
                    episode_lengths.append(i + 1)
                    break

                state = next_state
        return episode_lengths

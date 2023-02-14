'''Gridworld environments for experiments.'''

from typing import Tuple
import numpy as np

class GridWorld2Room(object):
    '''Class to represent a 2 room grid world environment'''

    def __init__(self, setup='default', start=None, goal=None) -> None:
        self.rows = 7
        self.cols = 9
        self._actions = [0, 1, 2, 3]
        self._action_strs = ['←', '↑', '→', '↓']
        self.curr_state = None
        self.start_state = None
        self.goal_state = None
        self.done = False
        self.blocked_states = [4, 13, 22, 40, 49, 58]

        # Determine what kind of environment this is going to be
        # Speocifically, how the start and goal states are determined
        if setup == 'default':
            self.set_default_start_goal_states()
        elif setup == 'random':
            self.set_random_start_goal_states()
        elif setup == 'custom':
            self.start_state = start
            self.goal_state = goal
            self.curr_state = self.start_state

    def actions(self) -> Tuple[int]:
        '''Return a list of available actions'''
        return self._actions

    def states(self) -> Tuple[int]:
        '''Return a list of possible states'''
        states = [s for s in range(self.rows * self.cols - 1)]# if s not in self.blocked_states]
        return states

    def set_default_start_goal_states(self) -> None:
        '''Set start and goal state to a default value'''
        self.start_state = 0
        self.curr_state = 0
        self.goal_state = self.rows * self.cols - 1

    def set_random_start_goal_states(self) -> None:
        '''Set start and goal states to random values, ensuring they are in different rooms'''
        total_states = (self.rows) * (self.cols) - 1
        s = np.random.randint(0, total_states + 1)
        g = np.random.randint(0, total_states + 1)

        # Ensure the states are not part of the wall in the middle of the room
        # And the start and end states are in different rooms (so that the problem is too easy)
        while s in self.blocked_states or g in self.blocked_states or self._same_room(s, g):
            s = np.random.randint(0, total_states + 1)
            g = np.random.randint(0, total_states + 1)
        self.start_state = s
        self.goal_state = g
        self.curr_state = self.start_state

    def reset(self) -> int:
        '''Resets the environment to the default setup.'''
        if self.start_state is None:
            self.set_default_start_goal_states()
        if isinstance(self.start_state, list):
            self.curr_state = np.random.choice(self.start_state)
        else:
            self.curr_state = self.start_state
        return self.curr_state

    def _same_room(self, start, goal) -> bool:
        '''Checks if the start and goal state are in the same room'''
        if start == goal:
            return True
        start_col = start % self.cols
        goal_col = goal % self.cols
        if (start_col < 4 and goal_col > 4) or (start_col > 4 and goal_col < 4):
            return False
        return True

    def neighbors(self, s) -> Tuple[int]:
        '''Find the neighbors of a give state. Uses the step function to explore'''
        # This is used for interpolation of the initiation set as mentioned in the paper
        # Included for completeness, the method works fine without this
        t = self.curr_state
        neighbors = []
        # basic actions (neighbors in top/bottom/left/right directions)
        for a in self._actions:
            self.curr_state = s
            ns, _ = self.step(a)
            neighbors.append(ns)
        # neighbors in diagonal directions
        for i in [1, 3]:
            for j in [0, 2]:
                self.curr_state = s
                _ = self.step(i)
                ns2, _ = self.step(j)
                neighbors.append(ns2)

        self.curr_state = t
        return list(set(neighbors))

    def step(self, action) -> Tuple[int, float, bool]:
        '''Execute the action in the environment'''
        if self.curr_state is None:
            return None
        if action not in self._actions:
            return self.curr_state
        new_state = self.curr_state
        if action == 0:     # left
            if self.curr_state % (self.cols) > 0:
                new_state = self.curr_state - 1
        elif action == 1:   # Up
            if self.curr_state >= self.cols:
                new_state = self.curr_state - self.cols
        elif action == 2:   # Right
            if self.curr_state % self.cols < (self.cols - 1):
                new_state = self.curr_state + 1
        elif action == 3:    # Down
            if self.curr_state < (self.rows - 1) * (self.cols):
                new_state = self.curr_state + self.cols

        # if colliding with the wall in the middle, dont move
        if new_state in self.blocked_states:
            new_state = self.curr_state

        self.curr_state = new_state
        if new_state == self.goal_state:
            return new_state, 1.0, True
        return new_state, 0.0, False

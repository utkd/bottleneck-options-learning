'''Utility functions'''

import numpy as np
from matplotlib import pyplot as plt

def plot_qtable(q_table, env):
    '''Plot the env grid and color coded values of greedy Q values'''
    values = np.zeros((env.rows)*(env.cols), )
    for s in q_table.keys():
        values[s] = np.max(q_table[s])
    for s in env.blocked_states:
        values[s] = -1
    values = values.reshape(env.rows, env.cols)
    plt.imshow(values,cmap='bone')
    plt.show()
    
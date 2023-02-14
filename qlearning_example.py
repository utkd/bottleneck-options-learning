'''File showing sample usage of basic Q-learning algorithm'''

from envs import GridWorld2Room
from qlearning import Qlearning
from utils import plot_qtable

if __name__ == "__main__":
    # env = GridWorld2Room(setup='random')
    env = GridWorld2Room(setup='custom', start=0, goal=62)
    qlearner = Qlearning(env)

    episode_lengths = qlearner.train(episodes=50, eps=0.1)
    print(episode_lengths)

plot_qtable(qlearner.q_table, env)
print(qlearner.greedy_rollout(env)[1])

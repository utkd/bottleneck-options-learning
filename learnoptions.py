'''Main script to learn a policy that uses options'''

from collections import defaultdict
import numpy as np
from matplotlib import pyplot as plt
from envs import GridWorld2Room
from qlearning import Qlearning
from options import generate_options
from options import choose_action

NUM_INITIAL_RANDOM_RUNS = 20    # Number of explorations to perform (start,goal) pairs
N_TRAIN = 200                   # Number of episodes in each exploration
N_OPTION_TRAIN = 200            # Number of episodes for learning oprion policies
N_TEST = 10                     # Number of test rollouts to perform after options have been learned
NUM_OPTIONS = 5                 # Number of options to find
N_POLICY_ON_OPTIONS_TRAIN = 50  # Number of episodes for learning with options + primitive actions

EPS = 0.1
LR = 0.1
GAMMA = 0.9

np.random.seed(1337)

# Generate State Visitation Counts

trajectories = []
print("Exploring ..")
for run_idx in range(NUM_INITIAL_RANDOM_RUNS):

    # Generate environments with random start/goal configs and explore
    env = GridWorld2Room(setup='random')
    agent = Qlearning(env)
    agent.train(episodes=N_TRAIN, lr=0.1, gamma=0.9, eps=0.1)

    for _ in range(N_TEST):
        env.reset()
        states, actions = agent.greedy_rollout(env)
        trajectories.append(states)


# Generate options and learn their policies
env = GridWorld2Room()
options = generate_options(trajectories, env, NUM_OPTIONS, False)

print("Learning Option Policies ..")
for opt in options:
    env = GridWorld2Room(setup='custom', start=opt.initiation_set, goal=opt.goal_state)
    opt.learn_option_policy(env, episodes=N_OPTION_TRAIN, lr=0.1, gamma=0.9, eps=0.4)


# Learn a policy over options
fig, ax = plt.subplots(2, 2)
print("Learning policy over options .. ")

# We train a policy with options on some sample start/goal states and compare
# with a policy learned through standard Q-learning
configs = [(0, 62), (17, 54), (9, 17), (61, 55)]     # (start,goal) states to learn for
for i_num, (start,goal) in enumerate(configs):

    # For this start and goal configuration, run Q-learning with options
    env = GridWorld2Room(setup='custom', start=start, goal=goal)
    all_actions = env.actions()
    all_actions.extend([i + len(env.actions()) for i in range(len(options))])

    q_table = defaultdict(lambda: np.zeros(len(all_actions)))

    episode_lengths_w_options = []

    for episode_num in range(N_POLICY_ON_OPTIONS_TRAIN):
        state = env.reset()

        episode_len = 0
        while True:
            episode_len += 1
            action = choose_action(q_table, options, state, EPS)

            # for primitive actions, just step through the environment as usual
            # for options, execute the option until completion
            if action < 4:
                next_state, reward, done = env.step(action)
                episode_len += 1
            else:
                opt = options[action - 4]
                next_state, reward, done, steps, _ = opt.execute_option(env, state, GAMMA)
                episode_len += steps

            next_q_values = q_table[next_state]
            best_next_action = np.argmax(next_q_values)

            td_target = reward + GAMMA * q_table[next_state][best_next_action]
            td_delta = td_target - q_table[state][action]
            q_table[state][action] += LR * td_delta

            if done:
                episode_lengths_w_options.append(episode_len)
                break

            state = next_state

    # Run regular Q-learning without options, for comparison
    qlearner = Qlearning(env)
    episode_lengths = qlearner.train(episodes=N_POLICY_ON_OPTIONS_TRAIN, eps=EPS)

    # Plot the results of learning with options and without options
    r = int(i_num / 2)
    c = i_num % 2
    ax[r, c].plot(episode_lengths_w_options) 
    ax[r, c].plot(episode_lengths)
    ax[r, c].legend(['With Options', 'Without Options'])
    ax[r, c].set_xlabel('Episodes')
    ax[r, c].set_ylabel('Episode Length')
    ax[r, c].set_title('Start:%d Goal:%d' % (start, goal), fontsize = 10, x=0.5, y=0.9)

plt.show()

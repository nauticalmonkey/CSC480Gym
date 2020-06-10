import gym
import random

import pickle
import numpy as np
import heuristic as tester

from state_discretizer import discretize_state
from gym.envs.box2d.lunar_lander import demo_heuristic_lander

# s[0] is the horizontal coordinate
# s[1] is the vertical coordinate
# s[2] is the horizontal speed
# s[3] is the vertical speed
# s[4] is the angle
# s[5] is the angular speed
# s[6] 1 if first leg has contact, else 0
# s[7] 1 if second leg has contact, else 0

"""
Debug levels:
    0: Minimal debug info
        - Util after each run and average after all iterations
    1: Q-map update info
    2: Misc info
        - State space and Action space info
        - Min/max of observations by dimension
"""
DEBUG = 0
RENDER = 0

nummberofepisodes = 100000


def main():
    path = 'qmap.pickle'
    q_map = get_q_map(path)
    environment = gym.make('LunarLander-v2')  # create the game

    if DEBUG > 1:
        print('State space: ', environment.observation_space)
        print(environment.observation_space.low)
        print(environment.observation_space.high)
        print('Action space: ', environment.action_space)

    rewards = []
    observations = []

    policy = q_map_policy

    i = 0
    loop = True # my debugger is not as good so i cant break in
    while i < nummberofepisodes:
        i += 1
    # for _ in range(nummberofepisodes):
        # episode_reward = demo_heuristic_lander(environment, render=True)
        # rewards.append(episode_reward)
        environment.seed(0)
        episode_reward = 0
        state = environment.reset()
        while True:
            if RENDER:
                environment.render()
            action = policy(q_map, state)
            newstate, reward, done, info = environment.step(action)
            episode_reward += reward
            observations.append(newstate)
            if done:
                print('Reward: %d' % episode_reward)
                rewards.append(episode_reward)
                break

            if state is not None:
                update_q_map(q_map, state, action, reward, newstate)
            state = newstate

    save_q_map(path, q_map)
    print('Average reward: %.2f' % (sum(rewards) / len(rewards)))

    if DEBUG > 2:
        print('Max observations: ',
              (np.array(observations).max(initial=float('-inf'), axis=0)))
        print('Min observations: ',
              (np.array(observations).min(initial=float('inf'), axis=0)))


# returns a random action
def random_policy(_, __):
    return random.randint(0, 3)


# Returns an action for a state
def q_map_policy(q_map, state):
    if state is None:
        print("WTF OMG")
        return random.randint(0,3)  # If state unknown perform random action

    util_max = float('-inf')
    action = None
    for i in range(0, 4):
        key = state_to_int(state, i)
        if key not in q_map:
            q_map[key] = 0

        util = q_map[key]
        if util > util_max:
            util_max = util
            action = i
    variation = random.randint(0,10)
    if variation > 8:
        action = random.randint(0,3)

    return action


def get_q_map(path):
    try:
        with open(path, 'rb') as file:
            q_map = pickle.load(file)
    except FileNotFoundError:
        q_map = dict()
        # save_q_map(q_map, path)

    return q_map


def save_q_map(path, q_map):
    with open(path, 'wb') as file:
        pickle.dump(q_map, file)


def dim_0_to_max(value, min_val, max_val, step):
    return (value - min_val) * (max_val - min_val) / step


def state_to_int(state, action):
    x = int(round(dim_0_to_max(state[0], -1, 1, 0.2)))
    y = int(round(dim_0_to_max(state[1], -1, 1, 0.2)))
    x_sp = int(round(dim_0_to_max(state[2], -2, 2, 0.2)))
    y_sp = int(round(dim_0_to_max(state[3], -2, 2, 0.2)))
    ang = int(round(dim_0_to_max(state[4], -2, 2, 0.4)))
    ang_sp = int(round(dim_0_to_max(state[5], -5, 5, 1)))
    left = state[6]
    right = state[7]
    a = action

    return int((x
                + y * 11
                + x_sp * 11 * 11
                + y_sp * 11 * 11 * 21
                + ang * 11 * 11 * 21 * 21
                + ang_sp * 11 * 11 * 21 * 21 * 11
                + left * 11 * 11 * 21 * 21 * 11 * 11
                + right * 11 * 11 * 21 * 21 * 11 * 11 * 2
                + a * 11 * 11 * 21 * 21 * 11 * 11 * 2 * 2))


# state you came from, action you took, reward that you got from doing it
def update_q_map(q_map, state, action, reward, newstate):
    learn_rate = 0.3
    discount = .8
    key = state_to_int(discretize_state(state), action)
    if key not in q_map:
        q_map[key] = 0

    util = q_map[key]
    new_util = util*(1-learn_rate) + learn_rate * (reward + discount *
                                    future_max(q_map,
                                               discretize_state(
                                                   newstate)) - util)

    if DEBUG > 2:
        print('State:', state, ', Action: ', action)
        print('Old util:', util)
        print('New util:', new_util)
    if DEBUG > 1:
        print('Delta: ', new_util - util)
    q_map[key] = new_util


def future_max(q_map, state):
    f_max = float('-inf')
    for i in range(0, 4):
        key = state_to_int(state, i)
        if key not in q_map:
            q_map[key] = 0

        f_util = q_map[key]
        if f_util > f_max:
            f_max = f_util
    return f_max


if __name__ == '__main__':
    main()

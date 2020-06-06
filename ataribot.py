import gym
import random

import pickle
import numpy as np

from state_discretizer import discretize_state
import itertools
from gym.envs.box2d.lunar_lander import demo_heuristic_lander

# s[0] is the horizontal coordinate
# s[1] is the vertical coordinate
# s[2] is the horizontal speed
# s[3] is the vertical speed
# s[4] is the angle
# s[5] is the angular speed
# s[6] 1 if first leg has contact, else 0
# s[7] 1 if second leg has contact, else 0


random.seed(52)

nummberofepisodes = 20


def main():
    q_map = get_q_map('qmap.pickle')
    environment = gym.make('LunarLander-v2')  # create the game

    print('State space: ', environment.observation_space)
    print(environment.observation_space.low)
    print(environment.observation_space.high)
    print('Action space: ', environment.action_space)

    environment.seed(52)

    rewards = []
    observations = []
    disc_obs = []

    # # action = 0
    for _ in range(nummberofepisodes):
        # episode_reward = demo_heuristic_lander(environment, render=True)
        # rewards.append(episode_reward)


        environment.reset()
        episode_reward = 0
        state = None
        while True:
            environment.render()
            # action = policy_function(q_map, state)
            action = environment.action_space.sample()
            newstate, reward, done, info = environment.step(action)  # preform random action
            episode_reward += reward
            observations.append(newstate)
            disc_obs.append(discretize_state(newstate))
            if done:
                print('Reward: %d' % episode_reward)
                rewards.append(episode_reward)
                break

            #Update map here
            if state is not None:
                update_q_map(q_map, state, action, reward, newstate)
            state = newstate

    print('Average reward: %.2f' % (sum(rewards) / len(rewards)))
    print('Max observations: ', (np.array(observations).max(initial=float('-inf'), axis=0)))
    print('Min observations: ', (np.array(observations).min(initial=float('inf'), axis=0)))
    print('Max observations: ', (np.array(disc_obs).max(initial=float('-inf'), axis=0)))
    print('Min observations: ', (np.array(disc_obs).min(initial=float('inf'), axis=0)))


# Returns an action for a state
def policy_function(q_map, state):
    util_max = float('-inf')
    action = None
    for i in range(0, 4):
        util = q_map((state, i))
        if util > util_max:
            util_max = util
            action = i

    return action



def get_q_map(path):
    try:
        with open(path, 'r') as file:
            q_map = pickle.load(file)
    except FileNotFoundError:
        q_map = init_q_map()

    return q_map


def init_q_map():
    return {((x / 10, y / 10, x_sp / 10, y_sp / 10, ang / 10, ang_sp, l, r), a): 0
            for x in range(-10, 12, 2)
            for y in range(-10, 12, 2)
            for x_sp in range(-20, 22, 2)
            for y_sp in range(-20, 22, 2)
            for ang in range(-20, 22, 4)
            for ang_sp in range(-5, 6)
            for l in {0, 1}
            for r in {0, 1}
            for a in range(0, 4)}

#state you came from, action you took, reward that you got from doing it
def update_q_map(q_map, state, action, reward, newstate):
    learning_rate = .5
    discount_factor = .8
    val_map = (tuple(discretize_state(state)), action)
    q_map[val_map] = q_map[val_map] + learning_rate * (reward + discount_factor * future_max(q_map, newstate) - q_map(val_map))


def future_max(q_map, state):
    f_max = float('-inf')
    for i in range(0, 4):
        if q_map(state, i) > f_max:
            f_max = q_map(state, i)
    return f_max




# def q_function(q_map, state, action, reward):
#     """
#     Maps a state and an action to a utility. To get the utility function, given
#     a state, simply get the max across all actions. To get the policy, return
#     that action.
#     :param state: the current state
#     :param action: the action to take
#     :param reward: The reward of taking the action given the current state (i.e. the resultant state)
#     :return: the utility for a given <state, action> pair
#     """
#
#     map_key = (tuple(discretize_state(state)), action)
#
#     # Q(s, a) <- (1 - learning_rate) * Q(s, a) + learning_rate(reward + discounting_factor * max{Q(s', a')})
#     learning_rate = 0.5
#     discount_factor = 0.8
#     old_q_value = q_map[map_key]
#
#
#     q_map
#
#     # q_function = {
#     #   (state, action): utility
#     # }
#
#
#     # 1. Initialize all (state,action) pairs to a abitrary fixed value
#     # 2. Iteratively update q_function until convergence (epsilon < ?)
#
#     return q_func





def policy(state):
    # argmax_a{Q(s,a)}
    pass


if __name__ == '__main__':
    main()
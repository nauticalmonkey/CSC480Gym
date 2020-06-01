import gym
import random
random.seed(52)

nummberofepisodes = 20


def main():
    environment = gym.make('SpaceInvaders-v0')  # create the game
    environment.seed(52)
    rewards = []

    for _ in range(nummberofepisodes):
        environment.reset()
        episode_reward = 0
        while True:
            action = environment.action_space.sample()
            _, reward, done, _ = environment.step(action)  # preform random action
            episode_reward += reward
            if done:
                print('Reward: %d' % episode_reward)
                rewards.append(episode_reward)
                break
    print('Average reward: %.2f' % (sum(rewards) / len(rewards)))


if __name__ == '__main__':
    main()
import pickle
import random
from argparse import ArgumentParser
from collections import deque
from typing import List, Optional

import gym
import numpy as np
from keras import Model, Sequential
from keras.activations import relu, linear
from keras.layers import Dense
from keras.optimizers import adam


class DQN:
    """
    A deep-Q-network based on Google DeepMind's architecture in
    "Human-level control through deep reinforcement learning"
    (Mnih et. al. 2015). Uses an epsilon-greedy policy for selecting actions and
    trains by "remembering" states visited and replaying a sample of memory
    on each iteration.
    """

    def __init__(self, action_space, state_space):
        # Used for determining number of nodes in input/output layers
        self.action_space = action_space
        self.state_space = state_space

        # Model hyper-parameters
        self.epsilon = 1.0
        self.epsilon_decay = .995
        self.epsilon_min = .1
        self.discount = .99
        self.batch_size = 32
        self.learning_rate = 0.001

        # Implement memory as a deque (fast push/pop list)
        self.memory = deque(maxlen=1000000)

        self.model = self.build_model()

    def build_model(self) -> Model:
        """
        Builds the neural network that will be used to approximate the
        q-function. The act method below will utilize the model to create the
        action policy, predicting the utility for all actions from a given state
        and choosing the action with the best predicted utility.

        :return: A neural network with a single hidden layer.
        """

        model = Sequential()
        model.add(Dense(150, input_dim=self.state_space, activation=relu))
        model.add(Dense(120, activation=relu))
        model.add(Dense(self.action_space, activation=linear))
        model.compile(loss='mse', optimizer=adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done) -> None:
        """
        Add a new entry to model's memory.

        :param state: Current state from which action is taken.
        :param action: Action that transitions from state -> next_state.
        :param reward: Reward of next_state.
        :param next_state: Resultant state of applying action to state.
        :param done: Indicator if episode is complete.
        """

        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, evaluate: bool = False) -> np.ndarray:
        """
        Enact policy function. Takes an epsilon greedy approach, starting with
        a high chance exploration and slowly transitioning to a greedy approach
        (exploitation) as we gain more data (and the reliability of the network
        increases).

        :param state: State from which an action needs to be taken.
        :param evaluate: Whether or not mode is being evaluated. 
        :return: The action to take.
        """

        if not evaluate and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self) -> None:
        """
        Replay, or learning phase, updating the q-function approximator by the
        following process:
            1. Sample a minibatch from memory and extract components.

            2. Get the target predictions. This is the portion of the update
            that is multiplied by the learning rate and looks as follows:

            reward + discount * max(Q(s', a')) e.g. estimate of future reward

            3. Get current predictions. Since the network predicts across
            all actions for a state we need the full set of predictions.

            4. Update the current predictions with the new target predictions
            for the action taken.

            5. Fit the model to the updated predictions.

            6. Lower randomness component of epsilon-greedy policy.
        """

        if len(self.memory) < self.batch_size:
            return

        # Step 1: Get minibatch
        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])
        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        # Step 2: Get target predictions
        targets = (rewards +
                   self.discount *
                   np.amax(self.model.predict_on_batch(next_states), axis=1) *
                   (1 - dones))  # If done, ignore "future" reward

        # Step 3: Get current predictions
        all_utils = self.model.predict_on_batch(states)
        indices = np.array([i for i in range(self.batch_size)])

        # Step 4: Update predictions
        all_utils[[indices], [actions]] = targets

        # Step 5: Fit the model
        self.model.fit(states, all_utils, epochs=1, verbose=0)

        # Step 6: Update policy
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filepath) -> None:
        """
        Save the DQN (Including NN and epsilon state.

        :param filepath: Location to save DQN to.
        """

        with open(filepath, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filepath: str) -> 'DQN':
        """
        Load a saved DQN from disk.

        :param filepath: The path to the model.
        :return: The loaded DQN model.
        """

        with open(filepath, 'rb') as file:
            return pickle.load(file)

    @staticmethod
    def get_agent(env: gym.Env, filepath: Optional[str]) -> 'DQN':
        """
        Get an agent from the given filepath if specified, otherwise construct a
        new model given the environment.

        :param env: The environment for constructing a new DQN
        :param filepath: The filepath to the DQN to load
        :return: A DQN model.
        """

        if filepath is None:
            return DQN(env.action_space.n, env.observation_space.shape[0])

        return DQN.load(filepath)


def run_dqn(agent: DQN, env: gym.Env, episodes: int, evaluate: bool,
            save_path: Optional[str]) -> List[float]:
    save = False  # Flags for altering execution control when debugging
    render = False
    state_space_shape = env.observation_space.shape[0]
    rewards = []

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, (1, state_space_shape))
        episode_reward = 0
        max_steps = 3000  # Guarantee episode termination
        for i in range(max_steps):
            action = agent.act(state, evaluate)
            if render or evaluate:
                env.render()
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            next_state = np.reshape(next_state, (1, state_space_shape))
            if not evaluate:
                agent.remember(state, action, reward, next_state, done)
                agent.replay()

            state = next_state
            if done:
                print(f"episode: {e + 1}/{episodes}, score: {episode_reward}")
                break
        rewards.append(episode_reward)

        if save and save_path is not None:
            agent.save(save_path)

        # Average score of last 100 episode, if it is over 200, we win!
        is_solved = np.mean(rewards[-100:])
        eval_size = min(e + 1, 100)
        print(f"Average over last {eval_size} episode: {is_solved:.2f}")

        if eval_size >= 100 and is_solved > 200:
            print('Game Solved!')

    if not evaluate and save_path is not None:
        agent.save(save_path)

    return rewards


def build_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Train or evaluate a deep-queue '
                                        'network on LunarLander.')
    parser.add_argument('mode', choices=['train', 'eval'],
                        help='Specify whether to run the agent in training '
                             'or evaluation mode')
    parser.add_argument('-n', '--number', type=int, default=100,
                        help='Number of iterations to run the agent')
    parser.add_argument('-f', '--filepath', type=str, default=None,
                        help='The path to save/load the agent pickle')
    return parser


def main() -> None:
    env = gym.make('LunarLander-v2')
    # env.seed(0)
    # np.random.seed(0)

    parser = build_arg_parser()
    args = parser.parse_args()

    filepath = args.filepath
    episodes = args.number
    evaluate = args.mode == 'eval'

    agent = DQN.get_agent(env, filepath)
    run_dqn(agent, env, episodes, evaluate, filepath)


if __name__ == '__main__':
    main()

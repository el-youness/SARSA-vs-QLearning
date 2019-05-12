import gym
from gym import wrappers, logger

import numpy as np

gym.logger.set_level(50)
env = gym.make('FrozenLake-v0')

min_epsilon = 0.1
max_epsilon = 1.0
decay_rate = 0.01


Q = np.zeros((env.observation_space.n, env.action_space.n))



def choose_action(state, epsilon):
    action = 0
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :])
    return action


def learn(state, state2, reward, action, lr_rate, gamma):
    predict = Q[state, action]
    target = reward + gamma * np.max(Q[state2, :])
    Q[state, action] = Q[state, action] + lr_rate * (target - predict)


def algo(epsilon, total_episodes, max_steps, lr_rate, gamma):
    rewards = 0
    # Start
    for episode in range(total_episodes):
        state = env.reset()
        t = 0

        while t < max_steps:
            action = choose_action(state, epsilon)

            state2, reward, done, info = env.step(action)

            learn(state, state2, reward, action, lr_rate, gamma)

            state = state2

            t += 1

            #You receive a reward of 1 if you reach the goal, and zero otherwise.
            rewards += reward

            if done:
                break
            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

    score = rewards / total_episodes
    print('played',total_episodes)
    print('won',rewards)
    print('score',score)


import gym
import numpy as np
import os
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


def learn(state, state2, reward, action, action2, lr_rate, gamma):
    predict = Q[state, action]
    target = reward + gamma * Q[state2, action2]
    Q[state, action] = Q[state, action] + lr_rate * (target - predict)

def algo(epsilon, total_episodes, max_steps, lr_rate, gamma):
    # Start
    rewards = 0

    for episode in range(total_episodes):
        t = 0
        state = env.reset()
        action = choose_action(state, epsilon)

        while t < max_steps:
            state2, reward, done, info = env.step(action)

            action2 = choose_action(state2, epsilon)

            learn(state, state2, reward, action, action2, lr_rate, gamma)

            state = state2

            action = action2

            t += 1
            rewards += reward

            if done:
                break
            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)


    score = rewards / total_episodes
    print('played', total_episodes)
    print('won', rewards)
    print('score', score)

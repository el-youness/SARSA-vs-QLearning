import gym
import numpy as np

env = gym.make('FrozenLake-v0')
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
            env.render()

            action = choose_action(state, epsilon)

            state2, reward, done, info = env.step(action)

            learn(state, state2, reward, action, lr_rate, gamma)

            state = state2

            t += 1

            rewards += reward

            if done:
                break
                
    score = rewards / total_episodes

    return Q
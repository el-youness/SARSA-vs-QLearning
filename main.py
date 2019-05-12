import sarsa
import qlearning
import numpy as np


def main():
    epsilon = 0.1
    total_episodes = 30000
    max_steps = 100

    lr_rate = 0.01
    gamma = 0.96

    algos_f = {
        "1" : qlearning,
        "2" : sarsa,
    }

    algos_f["2"].algo(epsilon, total_episodes, max_steps, lr_rate, gamma)
    '''
    for total_episodes in [1000,5000,10000,50000,100000]:
        print('total episodes', total_episodes)
        # Here we chose an algo
        algos_f["1"].algo(epsilon, total_episodes, max_steps, lr_rate, gamma)
        print()'''


if __name__ == "__main__":
    main()
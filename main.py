import sarsa
import qlearning
import numpy as np


def main():
    epsilon = 0.5
    total_episodes = 1000000
    max_steps = 1000

    lr_rate = 0.01
    gamma = 0.96
    activ_decay = True

    algos_f = {
        "1" : qlearning,
        "2" : sarsa,
    }


    algos_f["1"].algo(epsilon, total_episodes, max_steps, lr_rate, gamma, activ_decay)
    '''
    for total_episodes in [1000,5000,10000,50000,100000]:
        print('total episodes', total_episodes)
        # Here we chose an algo
        algos_f["1"].algo(epsilon, total_episodes, max_steps, lr_rate, gamma)
        print()'''


if __name__ == "__main__":
    main()
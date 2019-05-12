import sarsa
import qlearning



def main():
    epsilon = 0.5
    total_episodes = 100000
    max_steps = 100

    lr_rate = 0.1
    gamma = 0.96

    algos_f = {
        "1" : qlearning,
        "2" : sarsa,
    }

    # Here we chose an algo
    Q = algos_f["1"].algo(epsilon, total_episodes, max_steps, lr_rate, gamma)


if __name__ == "__main__":
    main()
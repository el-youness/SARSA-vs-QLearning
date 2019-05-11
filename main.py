import sarsa
import qlearning
import pickle

def dump_Q_table(filename, qtable):
    with open(filename, 'wb') as f:
        pickle.dump(qtable, f)

def main():
    epsilon = 0.9
    total_episodes = 1
    max_steps = 100

    lr_rate = 0.81
    gamma = 0.96

    algos_f = {
        "1" : qlearning,
        "2" : sarsa,
    }

    # Here we chose an algo
    Q = algos_f["1"]()
    dump_Q_table()


if __name__ == "__main__":
    main()
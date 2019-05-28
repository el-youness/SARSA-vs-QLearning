# Articial Intelligence Project : Q-learning versus SARSA for solving FrozenLake

**The full analysis and comparison of these 2 algorithms is described in the `report.pdf` file**

`main.py` imports the `qlearning.py` and `sarsa.py` files that contains the algos.

To change the algo used, just change the number in the line:
`algos_f["1"].algo(epsilon, total_episodes, max_steps, lr_rate, gamma)`</br>
   - 1 for Q-learning
   - 2 for SARSA
   
 Change the value of the `active_decay` attribute to `True` to enable the decreasing epslilon. `False` to disable it.

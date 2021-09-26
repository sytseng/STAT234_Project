## STAT234_Project_Spring_2021
This is the repository for Harvard STAT234 - Sequential Decision Making in Spring 2021. 

## Team member
* Shih-Yi Tseng
* Laurence Calancea

## Description
In this project, we construct and test the performance of reinforcement learning and adversarial search AI that play Checkers.

We tested the performance of RL agents using various online learning algorithms (SARSA, SARSA-lambda, MLP-based semi gradient SARSA) and with augmented features for board states against tree search-based alpha-beta pruning agents in non-stationary game environemnts in which the game rules change. To further allow efficient learning, we developed an algorithm for automatic detection of sudden changes in the environment and dynamically adjust learning parameters. Besides, we also tested the robustness of online learning RL agents vs. tree search-based alpha-beta pruning agents in stochastic environments in which the moves could be stochastically perturbed. Our project was a proof-of-principle how online learning algorithms can promote efficient learning in both non-stationary and stochastic environments.

A final report of this project can be find in this [file](https://github.com/sytseng/STAT234_Project/blob/main/Learning_for_Checkers_AI_final_report.pdf).


## Acknowledgements
We adapted the game structure: Board class in game.py and Game state class & Game rule class in checkers.py, as well as several basic agents (key board agent, alpha-beta agent, simple Q learning agent and simple SARSA agent) in agents.py from the [checkers-AI project](https://github.com/VarunRaval48/checkers-AI) of [VarunRaval48](https://github.com/VarunRaval48) with proper modifications.

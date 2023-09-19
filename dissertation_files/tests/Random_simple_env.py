from matplotlib import pyplot as plt
from dissertation_files.agents.agent import RandomAgent
from dissertation_files.environments.minigrid_environments import SimpleEnv
from dissertation_files.environments.minigrid_wrappers import FlatObsWrapper
from dissertation_files.agents.training import random_play_loop
from dissertation_files.agents.evaluation import plot_evaluation_data

"""
## Hyperparameters
"""

STEPS_PER_EPOCH = 4000
EPOCHS = 30
EVALUATION_FREQUENCY = 2

"""
## Run
"""

env = SimpleEnv(render_mode=None)
env = FlatObsWrapper(env)
action_dimensions = env.action_space.n

agent = RandomAgent(action_dimensions)

average_reward_list = random_play_loop(EPOCHS, agent, env, STEPS_PER_EPOCH)
average_reward_list = average_reward_list[0:len(average_reward_list)//EVALUATION_FREQUENCY]

plot_evaluation_data([average_reward_list], EPOCHS, EVALUATION_FREQUENCY, STEPS_PER_EPOCH)



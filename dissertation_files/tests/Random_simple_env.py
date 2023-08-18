from matplotlib import pyplot as plt
from dissertation_files.agents.agent import RandomAgent
from dissertation_files.environments.simple_env import SimpleEnv
from dissertation_files.environments.minigrid_wrappers import FlatObsWrapper
from dissertation_files.agents.training import random_play_loop

"""
## Hyperparameters
"""

STEPS_PER_EPOCH = 4000
EPOCHS = 30

"""
## Run
"""

env = SimpleEnv(render_mode=None)
env = FlatObsWrapper(env)
action_dimensions = env.action_space.n

agent = RandomAgent(action_dimensions)

average_reward_list = random_play_loop(EPOCHS, agent, env, STEPS_PER_EPOCH)

reward_plot = plt.plot([i+1 for i in range(EPOCHS)], average_reward_list)
plt.show()


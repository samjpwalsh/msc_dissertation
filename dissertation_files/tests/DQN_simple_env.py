from keras import activations
from matplotlib import pyplot as plt
from dissertation_files.environments.simple_env import SimpleEnv
from dissertation_files.agents.agent import DQNAgent
from dissertation_files.environments.minigrid_wrappers import FlatObsWrapper
from dissertation_files.agents.training import dqn_training_loop


"""
Hyperparameters
"""

EPISODES = 500
BATCH_SIZE = 50
MEMORY_SIZE = 10000
GAMMA = 0.95
EPSILON = 1.0
MIN_EPSILON = 0.01
EPSILON_DECAY = 0.99
LEARNING_RATE = 0.001
STEPS_TARGET_MODEL_UPDATE = 100
HIDDEN_SIZES = (64, 64)
INPUT_ACTIVATION = activations.relu
OUTPUT_ACTIVATION = None

"""
## Run
"""

env = SimpleEnv(render_mode=None)
env = FlatObsWrapper(env)
observation_dimensions = len(env.reset()[0])
action_dimensions = env.action_space.n
agent = DQNAgent(observation_dimensions, action_dimensions, MEMORY_SIZE, BATCH_SIZE,
                 HIDDEN_SIZES, INPUT_ACTIVATION, OUTPUT_ACTIVATION, LEARNING_RATE,
                 EPSILON, EPSILON_DECAY, MIN_EPSILON, GAMMA)

reward_list = dqn_training_loop(EPISODES, agent, env, observation_dimensions, STEPS_TARGET_MODEL_UPDATE)

reward_plot = plt.plot([i+1 for i in range(EPISODES)], reward_list)
plt.show()

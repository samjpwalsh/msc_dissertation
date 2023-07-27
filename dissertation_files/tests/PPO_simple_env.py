from keras import activations
from matplotlib import pyplot as plt
from dissertation_files.agents.agent import PPOAgent
from dissertation_files.environments.simple_env import SimpleEnv
from dissertation_files.environments.minigrid_wrappers import FlatObsWrapper
from dissertation_files.agents.training import ppo_training_loop


"""
## Hyperparameters
"""

STEPS_PER_EPOCH = 4000
EPOCHS = 30
GAMMA = 0.99
CLIP_RATIO = 0.2
ACTOR_LEARNING_RATE = 3e-4
CRITIC_LEARNING_RATE = 1e-3
TRAIN_ACTOR_ITERATIONS = 80
TRAIN_CRITIC_ITERATIONS = 80
LAM = 0.97
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

agent = PPOAgent(observation_dimensions, action_dimensions, STEPS_PER_EPOCH, HIDDEN_SIZES, INPUT_ACTIVATION,
                 OUTPUT_ACTIVATION, ACTOR_LEARNING_RATE, CRITIC_LEARNING_RATE, CLIP_RATIO, GAMMA, LAM)

episodes, reward_list = ppo_training_loop(EPOCHS, agent, env, observation_dimensions, action_dimensions,
                                          STEPS_PER_EPOCH, TRAIN_ACTOR_ITERATIONS, TRAIN_CRITIC_ITERATIONS)

reward_plot = plt.plot([i+1 for i in range(episodes)], reward_list)
plt.show()

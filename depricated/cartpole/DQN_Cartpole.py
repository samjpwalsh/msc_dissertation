import gymnasium as gym
from keras import activations
from dissertation_files.agents.agent import DQNAgent
from dissertation_files.agents.training import dqn_training_loop
from dissertation_files.agents.evaluation import plot_evaluation_data


"""
Hyperparameters
"""

STEPS_PER_EPOCH = 4000
EPOCHS = 3
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
EVALUATION_FREQUENCY = 2
EVALUATION_EPISODES_PER_EPOCH = 10

"""
## Run
"""

env = gym.make('CartPole-v1')
eval_env = gym.make('CartPole-v1', render_mode=None)
observation_dimensions = env.observation_space.shape[0]
action_dimensions = env.action_space.n
agent = DQNAgent(observation_dimensions, action_dimensions, MEMORY_SIZE, BATCH_SIZE,
                 HIDDEN_SIZES, INPUT_ACTIVATION, OUTPUT_ACTIVATION, LEARNING_RATE,
                 EPSILON, EPSILON_DECAY, MIN_EPSILON, GAMMA)

average_reward_list, first_time_visits = dqn_training_loop(EPOCHS, agent, env, observation_dimensions, STEPS_PER_EPOCH,
                                        STEPS_TARGET_MODEL_UPDATE, eval_env=eval_env,
                                        eval_epoch_frequency=EVALUATION_FREQUENCY,
                                        eval_episodes_per_epoch=EVALUATION_EPISODES_PER_EPOCH)

plot_evaluation_data([average_reward_list], EPOCHS, EVALUATION_FREQUENCY, STEPS_PER_EPOCH)


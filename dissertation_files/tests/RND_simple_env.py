from keras import activations
from dissertation_files.agents.agent import RNDAgent
from dissertation_files.environments.simple_env import SimpleEnv
from dissertation_files.environments.minigrid_wrappers import FlatObsWrapper
from dissertation_files.agents.training import rnd_training_loop
from dissertation_files.agents.evaluation import plot_evaluation_data

"""
## Hyperparameters
"""

STEPS_PER_EPOCH = 4000
EPOCHS = 2
GAMMA = 0.99
CLIP_RATIO = 0.2
ACTOR_LEARNING_RATE = 3e-4
CRITIC_LEARNING_RATE = 1e-3
RND_PREDICTOR_LEARNING_RATE = 3e-4
TRAIN_ACTOR_ITERATIONS = 80
TRAIN_CRITIC_ITERATIONS = 80
TRAIN_RND_ITERATIONS = 80
LAM = 0.97
HIDDEN_SIZES = (64, 64)
INPUT_ACTIVATION = activations.relu
OUTPUT_ACTIVATION = None
EVALUATION_FREQUENCY = 2
EVALUATION_EPISODES_PER_EPOCH = 1

"""
## Run
"""

env = SimpleEnv(render_mode=None)
env = FlatObsWrapper(env)
eval_env = SimpleEnv(render_mode=None)
eval_env = FlatObsWrapper(eval_env)
observation_dimensions = len(env.reset()[0])
action_dimensions = env.action_space.n

agent = RNDAgent(observation_dimensions, action_dimensions, STEPS_PER_EPOCH, HIDDEN_SIZES, INPUT_ACTIVATION,
                 OUTPUT_ACTIVATION, ACTOR_LEARNING_RATE, CRITIC_LEARNING_RATE, RND_PREDICTOR_LEARNING_RATE,
                 CLIP_RATIO, GAMMA, LAM)

average_reward_list, average_intrinsic_reward_list = rnd_training_loop(EPOCHS, agent, env, observation_dimensions,
                                                                       action_dimensions, STEPS_PER_EPOCH,
                                                                       TRAIN_ACTOR_ITERATIONS, TRAIN_CRITIC_ITERATIONS,
                                                                       TRAIN_RND_ITERATIONS, eval_env=eval_env,
                                                                       eval_epoch_frequency=EVALUATION_FREQUENCY,
                                                                       eval_episodes_per_epoch=EVALUATION_EPISODES_PER_EPOCH)

plot_evaluation_data([average_reward_list, average_intrinsic_reward_list], EPOCHS, EVALUATION_FREQUENCY, STEPS_PER_EPOCH)
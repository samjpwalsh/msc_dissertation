import numpy as np
from keras import activations
from matplotlib import pyplot as plt
from dissertation_files.environments.simple_env import SimpleEnv
from dissertation_files.agents.agent import DQNAgent
from dissertation_files.environments.minigrid_wrappers import FlatObsWrapper


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
HIDDEN_SIZES = (24, 24)
INPUT_ACTIVATION = activations.relu
OUTPUT_ACTIVATION = None

"""
Initialisations
"""

env = SimpleEnv(render_mode=None)
env = FlatObsWrapper(env)
observation_dimensions = len(env.reset()[0])
action_dimensions = env.action_space.n
agent = DQNAgent(observation_dimensions, action_dimensions, MEMORY_SIZE, BATCH_SIZE,
                 HIDDEN_SIZES, INPUT_ACTIVATION, OUTPUT_ACTIVATION, LEARNING_RATE,
                 EPSILON, EPSILON_DECAY, MIN_EPSILON, GAMMA)
reward_list = []
step_counter = 0

"""
Training
"""

for episode in range(EPISODES):
    observation = env.reset()[0]
    observation = np.reshape(observation, [1, observation_dimensions])
    done = False
    truncated = False
    episode_reward = 0
    while not done:
        action = agent.sample_action(observation)
        next_observation, reward, done, truncated, _ = env.step(action)
        if truncated:
            done = True
        next_observation = np.reshape(next_observation, [1, observation_dimensions])
        agent.buffer.store(observation, action, next_observation, reward, done)
        agent.train_model()
        if step_counter % STEPS_TARGET_MODEL_UPDATE == 0 and step_counter != 0:
            agent.update_target_model()
        episode_reward += reward
        observation = next_observation
        step_counter += 1
    print(f"episode: {episode+1}/{EPISODES}, score: {episode_reward}, steps: {step_counter}")
    reward_list.append(episode_reward)

reward_plot = plt.plot([i+1 for i in range(EPISODES)], reward_list)
plt.show()

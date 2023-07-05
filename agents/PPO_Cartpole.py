import numpy as np
from tensorflow import keras
import gymnasium as gym
from utils import logprobabilities
from agent import PPOAgent


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
INPUT_ACTIVATION = keras.activations.tanh
OUTPUT_ACTIVATION = None


"""
## Initializations
"""

env = gym.make("CartPole-v1", render_mode=None)
observation_dimensions = env.observation_space.shape[0]
action_dimensions = env.action_space.n

agent = PPOAgent(observation_dimensions, action_dimensions, STEPS_PER_EPOCH, HIDDEN_SIZES, INPUT_ACTIVATION,
                 OUTPUT_ACTIVATION, ACTOR_LEARNING_RATE, CRITIC_LEARNING_RATE, CLIP_RATIO, GAMMA, LAM)

observation, episode_return, episode_length = env.reset()[0], 0, 0

"""
## Training
"""

for epoch in range(EPOCHS):

    sum_return = 0
    sum_length = 0
    num_episodes = 0

    # Iterate over the steps of each epoch
    for t in range(STEPS_PER_EPOCH):

        if len(observation) == 2: # fix this
            observation = observation[0]
        observation = np.reshape(observation, [1, env.observation_space.shape[0]])
        logits, action = agent.sample_action(observation)
        observation_new, reward, done, _, _ = env.step(action[0].numpy())
        episode_return += reward
        episode_length += 1

        value_t = agent.critic(observation)
        logprobability_t = logprobabilities(logits, action, action_dimensions)

        agent.buffer.store(observation, action, reward, value_t, logprobability_t)

        observation = observation_new

        terminal = done
        if terminal or (t == STEPS_PER_EPOCH - 1):
            last_value = 0 if done else agent.critic(observation.reshape(1, -1))
            agent.buffer.finish_trajectory(last_value)
            sum_return += episode_return
            sum_length += episode_length
            num_episodes += 1
            observation, episode_return, episode_length = env.reset(), 0, 0

    (
        observation_buffer,
        action_buffer,
        advantage_buffer,
        return_buffer,
        logprobability_buffer,
    ) = agent.buffer.get()

    for _ in range(TRAIN_ACTOR_ITERATIONS):
        agent.train_actor(observation_buffer, action_buffer, logprobability_buffer, advantage_buffer)

    for _ in range(TRAIN_CRITIC_ITERATIONS):
        agent.train_critic(observation_buffer, return_buffer)

    print(
        f" Epoch: {epoch + 1}. Mean Return: {sum_return / num_episodes}. Mean Length: {sum_length / num_episodes}"
    )



## Aiming to refactor so all agents have their own class (seperate from the buffer classes), then maybe have a
## single "agent" python file

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras
from keras.models import clone_model
import gymnasium as gym
from buffer import DQNBuffer as Buffer
from utils import mlp
import random

"""
Agent Class
"""


class DQNAgent:
    def __init__(self, observation_dimensions, action_dimensions, memory_size, batch_size,
                 hidden_sizes, input_activation, output_activation, learning_rate,
                 epsilon, epsilon_decay, min_epsilon, gamma):
        self.observation_dimensions = observation_dimensions
        self.action_dimensions = action_dimensions
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.buffer = Buffer(observation_dimensions, memory_size, batch_size)
        self.model, self.optimizer = self.build_model(hidden_sizes, input_activation, output_activation)
        self.target_model = self.build_target_model()

    def build_model(self, hidden_sizes, input_activation, output_activation):
        observation_input = keras.Input(shape=(self.observation_dimensions,), dtype=tf.float32)
        logits = mlp(observation_input, list(hidden_sizes) + [self.action_dimensions], input_activation, output_activation)
        model = keras.Model(inputs=observation_input, outputs=logits)
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        return model, optimizer

    def build_target_model(self):
        target_model = clone_model(self.model)
        return target_model


    def sample_action(self, observation):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dimensions)
        else:
            return np.argmax(self.model(observation)[0])


    def train_model(self):
        if self.buffer.pointer < self.buffer.batch_size:
            return

        observations, actions, rewards, next_observations, dones = self.buffer.sample()

        targets = rewards.copy()
        not_done_mask = ~dones
        next_state_predictions = self.target_model(next_observations)
        Q_values_next = np.amax(next_state_predictions.numpy(), axis=1)
        targets[not_done_mask] += self.gamma * Q_values_next[not_done_mask]

        with tf.GradientTape() as tape:
            q_values = self.model(observations)
            q_values_actions = tf.reduce_sum(q_values * tf.one_hot(actions, self.action_dimensions), axis=1)
            loss = tf.reduce_mean(tf.square(targets - q_values_actions))

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.min_epsilon, self.epsilon)

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

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
INPUT_ACTIVATION = keras.activations.relu
OUTPUT_ACTIVATION = None

"""
Initialisations
"""

env = gym.make('CartPole-v1')
observation_dimensions = env.observation_space.shape[0]
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
    episode_reward = 0
    while not done:
        action = agent.sample_action(observation)
        next_observation, reward, done, _, _ = env.step(action)
        next_observation = np.reshape(next_observation, [1, observation_dimensions])
        agent.buffer.store(observation, action, next_observation, reward, done)
        agent.train_model()
        if step_counter % STEPS_TARGET_MODEL_UPDATE == 0 and step_counter != 0:
            agent.update_target_model()
        episode_reward += reward
        observation = next_observation
        step_counter += 1
        if done:
            print(f"episode: {episode+1}/{EPISODES}, score: {episode_reward}, steps: {step_counter}")
            reward_list.append(episode_reward)
    reward_plot = plt.plot([i+1 for i in range(episode+1)], reward_list)
    plt.show()

import gymnasium as gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
import gymnasium as gym
from itertools import count

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation
from keras.optimizers import Adam

# Define Replay Memory Class
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(*args)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Define Q network class

class DQN:
    def __init__(self, observation_size, action_size, gamma, epsilon, epsilon_decay, epsilon_min, batch_size):
        self.observation_size = observation_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.nn = Sequential([
            Dense(128, input_shape=self.observation_size, activation='relu'),
            Dense(128, activation='relu'),
            Dense(self.action_size, activation='softmax')])
        self.nn.compile(loss='huber', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
        self.replay_memory = ReplayMemory(1000)

    def select_action(self, state):
        if np.random.uniform() < self.epsilon:
            return env.action_space.sample()
        else:
            np.argmax(self.nn.predict(state))

    def update_model(self):
        if len(self.replay_memory) < self.batch_size:
            return

        memories = self.replay_memory.sample(self.batch_size)
        for memory in memories:
            s, a, r, s1, done = memory
            if not done:
                x = s1.shape
                target = r + self.gamma * np.amax(self.nn.predict((s1,)))
            else:
                target = r

            target_f = self.nn.predict((s,))
            target_f[0][a] = target
            self.nn.fit(s, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# Environment
env = gym.make("CartPole-v1")

# Hyperparameters

GAMMA = 0.95
EPSILON = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01
BATCH_SIZE = 32

# Train the model
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
dqn = DQN(state_size, action_size, GAMMA, EPSILON, EPSILON_DECAY, EPSILON_MIN, BATCH_SIZE)
episodes = 1000
reward_list = []

for episode in range(episodes):
    state = env.reset()[0]
    state = np.reshape(state, [1, state_size])
    done = False
    cumulative_reward = 0
    if episode % 10 == 0:
        env.render()
    while not done:
        action = dqn.select_action(state)
        next_state, reward, done, _, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        dqn.replay_memory.push((state, action, reward, next_state, done))
        state = next_state
        cumulative_reward += reward
        dqn.update_model()
    reward_list.append(cumulative_reward)

reward_plot = plt.plot(x=[i for i in range(episodes)], y=reward_list)
plt.show()















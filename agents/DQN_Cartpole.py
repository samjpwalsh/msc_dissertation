import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
from keras.models import Sequential, clone_model
from keras.layers import Dense
from keras.optimizers import Adam

EPISODES = 50
BATCH_SIZE = 32
MEMORY_SIZE = 100_000
GAMMA = 0.95
EPSILON = 1.0
EPSILON_DECAY = 0.99
MIN_EPSILON = 0.01
LEARNING_RATE = 0.001
STEPS_TARGET_MODEL_UPDATE = 1000

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON
        self.model = self.build_model()
        self.target_model = self.build_target_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='huber', optimizer=Adam(lr=LEARNING_RATE))
        return model

    def build_target_model(self):
        target_model = clone_model(self.model)
        target_model.compile(loss='huber', optimizer=Adam(lr=LEARNING_RATE))
        return target_model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.model.predict(state, verbose=0)[0])

    def update_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, next_state, done in batch:
            if done:
                target = reward
            else:
                target = reward + GAMMA * np.amax(self.target_model.predict(next_state, verbose=0)[0])
            prediction = self.model.predict(state, verbose=0)
            prediction[0][action] = target # ? investigate
            self.model.fit(state, prediction, epochs=1, verbose=0)
        if self.epsilon > MIN_EPSILON:
            self.epsilon *= EPSILON_DECAY

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())


if __name__ == "__main__":

    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    reward_list = []
    for episode in range(EPISODES):
        state = env.reset()[0]
        state = np.reshape(state, [1, state_size])
        done = False
        episode_reward = 0
        step_counter = 0
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            agent.update_model()
            if step_counter % STEPS_TARGET_MODEL_UPDATE == 0:
                agent.update_target_model()
            episode_reward += reward
            state = next_state
            step_counter += 1
            if done:
                print(f"episode: {episode+1}/{EPISODES}, score: {episode_reward}")
                reward_list.append(episode_reward)
    reward_plot = plt.plot([i+1 for i in range(EPISODES)], reward_list)
    plt.show()

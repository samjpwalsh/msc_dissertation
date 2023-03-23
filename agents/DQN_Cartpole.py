import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
from keras.models import Sequential, clone_model
from keras.layers import Dense
from keras.optimizers import Adam

EPISODES = 150
BATCH_SIZE = 20
MEMORY_SIZE = 10000
GAMMA = 0.95
EPSILON = 1.0
MIN_EPSILON = 0.01
EPSILON_DECAY = 0.99
LEARNING_RATE = 0.001
STEPS_TARGET_MODEL_UPDATE = 100

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = np.zeros(shape=MEMORY_SIZE, dtype=np.ndarray)
        self.memory_index = 0
        self.epsilon = EPSILON
        self.model = self.build_model()
        self.target_model = self.build_target_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=LEARNING_RATE))
        return model

    def build_target_model(self):
        target_model = clone_model(self.model)
        target_model.compile(loss='mse', optimizer=Adam(lr=LEARNING_RATE))
        return target_model

    def remember(self, state, action, reward, next_state, done):
        self.memory[self.memory_index] = (state, action, reward, next_state, done)
        self.memory_index = (self.memory_index + 1) % MEMORY_SIZE

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.model.predict(state, verbose=0)[0])

    def update_model(self):
        if self.memory_index < BATCH_SIZE:
            return
        indices = np.random.choice(len(np.nonzero(self.memory)[0]), BATCH_SIZE, replace=False)
        batch = self.memory[indices]

        states = np.array([item[0][0] for item in batch])
        actions = np.array([item[1] for item in batch])
        rewards = np.array([item[2] for item in batch])
        next_states = np.array([item[3][0] for item in batch])
        dones = np.array([item[4] for item in batch])

        targets = rewards.copy()
        not_done_mask = ~dones
        next_state_predictions = self.target_model.predict(next_states, verbose=0)
        Q_values_next = np.amax(next_state_predictions, axis=1)
        targets[not_done_mask] += GAMMA * Q_values_next[not_done_mask]

        predictions = self.model.predict(states, verbose=0)
        predictions[range(len(actions)), actions] = targets

        self.model.fit(states, predictions, epochs=1, verbose=0)
        self.epsilon *= EPSILON_DECAY
        self.epsilon = max(MIN_EPSILON, self.epsilon)

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())


if __name__ == "__main__":

    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    reward_list = []
    step_counter = 0
    for episode in range(EPISODES):
        state = env.reset()[0]
        state = np.reshape(state, [1, state_size])
        done = False
        episode_reward = 0
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            agent.update_model()
            if step_counter % STEPS_TARGET_MODEL_UPDATE == 0 and step_counter != 0:
                agent.update_target_model()
            episode_reward += reward
            state = next_state
            step_counter += 1
            if done:
                print(f"episode: {episode+1}/{EPISODES}, score: {episode_reward}, steps: {step_counter}")
                reward_list.append(episode_reward)
        reward_plot = plt.plot([i+1 for i in range(episode+1)], reward_list)
        plt.show()

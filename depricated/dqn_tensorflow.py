import gym
import numpy as np
import random
import tensorflow as tf

env = gym.make('CartPole-v1')

num_episodes = 200
batch_size = 32
memory_size = 100000
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
target_update_freq = 1000
learning_rate = 0.001


class ReplayBuffer():
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.buffer = []
        self.index = 0

    def add(self, experience):
        if len(self.buffer) < self.memory_size:
            self.buffer.append(experience)
        else:
            self.buffer[self.index] = experience
            self.index = (self.index + 1) % self.memory_size

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)


state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n


class DQNAgent():
    def __init__(self):
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.replay_buffer = ReplayBuffer(memory_size)
        self.epsilon = epsilon

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_dim=state_dim),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(action_dim, activation=None)
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                      loss='mse')
        return model

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return env.action_space.sample()
        else:
            return np.argmax(self.model.predict(np.array([state]))[0])


agent = DQNAgent()
global_step = 0
for episode in range(num_episodes):
    state = env.reset()[0]
    done = False
    episode_reward = 0

    while not done:
        action = agent.act(state)
        next_state, reward, done, _, _ = env.step(action)
        state = np.reshape(state, [1, state_dim])
        next_state = np.reshape(next_state, [1, state_dim])
        agent.replay_buffer.add((state, action, reward, next_state, done))
        episode_reward += reward
        state = next_state
        global_step += 1

        if len(agent.replay_buffer.buffer) >= batch_size:
            batch = agent.replay_buffer.sample(batch_size)
            states = np.array([item[0][0] for item in batch])
            actions = np.array([item[1] for item in batch])
            rewards = np.array([item[2] for item in batch])
            next_states = np.array([item[3][0] for item in batch])
            dones = np.array([item[4] for item in batch])
            q_values = agent.model.predict(states, verbose=0)
            next_q_values = agent.target_model.predict(next_states, verbose=0)
            targets = np.array(q_values)
            not_done_indexes = np.where(dones == False)[0]
            next_state_predictions = agent.target_model.predict(next_states, verbose=0)
            Q_values_next = np.amax(next_state_predictions, axis=1)
            targets[not_done_indexes] += gamma * Q_values_next[not_done_indexes]
            agent.model.fit(states, targets, epochs=1, verbose=0)

        if global_step % target_update_freq == 0:
            agent.target_model.set_weights(agent.model.get_weights())

agent.epsilon = max(agent.epsilon * epsilon_decay, epsilon_min)
print("Episode:", episode, "Reward:", episode_reward)


import numpy as np
from dissertation_files.agents.utils import discounted_cumulative_sums


class DQNBuffer:

    def __init__(self, observation_dimensions, size, batch_size):
        # Buffer initialization
        self.observation_buffer = np.zeros(
            (size, observation_dimensions), dtype=np.float32
        )
        self.action_buffer = np.zeros(size, dtype=np.int32)
        self.next_observation_buffer = np.zeros(
            (size, observation_dimensions), dtype=np.float32
        )
        self.reward_buffer = np.zeros(size, dtype=np.float32)
        self.done_buffer = np.zeros(size, dtype=bool)
        self.pointer = 0
        self.size = size
        self.batch_size = batch_size
        self.buffer_full = False

    def store(self, observation, action, next_observation, reward, done):
        self.observation_buffer[self.pointer] = observation
        self.action_buffer[self.pointer] = action
        self.next_observation_buffer[self.pointer] = next_observation
        self.reward_buffer[self.pointer] = reward
        self.done_buffer[self.pointer] = done
        if self.pointer == (self.size - 1):
            self.buffer_full = True
        self.pointer = (self.pointer + 1) % self.size

    def sample(self):
        if self.buffer_full:
            indices = np.random.choice(range(self.size), self.batch_size, replace=False)
        else:
            indices = np.random.choice(range(self.pointer + 1), self.batch_size, replace=False)
        observations = self.observation_buffer[indices]
        actions = self.action_buffer[indices]
        rewards = self.reward_buffer[indices]
        next_observations = self.next_observation_buffer[indices]
        dones = self.done_buffer[indices]
        return observations, actions, rewards, next_observations, dones


class PPOBuffer:

    def __init__(self, observation_dimensions, size, gamma, lam):
        # Buffer initialization
        self.observation_buffer = np.zeros(
            (size, observation_dimensions), dtype=np.float32
        )
        self.action_buffer = np.zeros(size, dtype=np.int32)
        self.advantage_buffer = np.zeros(size, dtype=np.float32)
        self.reward_buffer = np.zeros(size, dtype=np.float32)
        self.return_buffer = np.zeros(size, dtype=np.float32)
        self.value_buffer = np.zeros(size, dtype=np.float32)
        self.logprobability_buffer = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.pointer, self.trajectory_start_index = 0, 0

    def store(self, observation, action, reward, value, logprobability):
        # Append one step of agent-environment interaction
        self.observation_buffer[self.pointer] = observation
        self.action_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward
        self.value_buffer[self.pointer] = value
        self.logprobability_buffer[self.pointer] = logprobability
        self.pointer += 1

    def finish_trajectory(self, last_value=0):
        # Finish the trajectory by computing advantage estimates and rewards-to-go
        path_slice = slice(self.trajectory_start_index, self.pointer)
        rewards = np.append(self.reward_buffer[path_slice], last_value)
        values = np.append(self.value_buffer[path_slice], last_value)

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        self.advantage_buffer[path_slice] = discounted_cumulative_sums(
            deltas, self.gamma * self.lam
        )
        self.return_buffer[path_slice] = discounted_cumulative_sums(
            rewards, self.gamma
        )[:-1]

        self.trajectory_start_index = self.pointer

    def get(self):
        # Get all data of the buffer and normalize the advantages
        self.pointer, self.trajectory_start_index = 0, 0
        advantage_mean, advantage_std = (
            np.mean(self.advantage_buffer),
            np.std(self.advantage_buffer),
        )
        self.advantage_buffer = (self.advantage_buffer - advantage_mean) / advantage_std
        return (
            self.observation_buffer,
            self.action_buffer,
            self.advantage_buffer,
            self.return_buffer,
            self.logprobability_buffer,
        )


class RNDBuffer:

    def __init__(self, observation_dimensions, size, gamma=0.99, lam=0.95):
        # Buffer initialization
        self.observation_buffer = np.zeros(
            (size, observation_dimensions), dtype=np.float32
        )
        self.action_buffer = np.zeros(size, dtype=np.int32)
        self.extrinsic_advantage_buffer = np.zeros(size, dtype=np.float32)
        self.intrinsic_advantage_buffer = np.zeros(size, dtype=np.float32)
        self.total_advantage_buffer = np.zeros(size, dtype=np.float32)
        self.extrinsic_reward_buffer = np.zeros(size, dtype=np.float32)
        self.intrinsic_reward_buffer = np.zeros(size, dtype=np.float32)
        self.total_reward_buffer = np.zeros(size, dtype=np.float32)
        self.value_buffer = np.zeros(size, dtype=np.float32)
        self.logprobability_buffer = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.pointer, self.trajectory_start_index = 0, 0

    def store(self, observation, action, extrinsic_reward, intrinsic_reward, value, logprobability):
        # Append one step of agent-environment interaction
        self.observation_buffer[self.pointer] = observation
        self.action_buffer[self.pointer] = action
        self.extrinsic_reward_buffer[self.pointer] = extrinsic_reward
        self.intrinsic_reward_buffer[self.pointer] = intrinsic_reward
        self.value_buffer[self.pointer] = value
        self.logprobability_buffer[self.pointer] = logprobability
        self.pointer += 1

    def finish_trajectory(self, last_value=0):
        path_slice = slice(self.trajectory_start_index, self.pointer)
        extrinsic_rewards = np.append(self.extrinsic_reward_buffer[path_slice], last_value)
        intrinsic_rewards = np.append(self.intrinsic_reward_buffer[path_slice], last_value)
        values = np.append(self.value_buffer[path_slice], last_value)

        # Extrinsic Rewards and Advantages
        ex_deltas = extrinsic_rewards[:-1] + self.gamma * values[1:] - values[:-1]
        self.extrinsic_advantage_buffer[path_slice] = discounted_cumulative_sums(
            ex_deltas, self.gamma * self.lam
        )
        self.extrinsic_reward_buffer[path_slice] = discounted_cumulative_sums(
            extrinsic_rewards, self.gamma
        )[:-1]

        # Intrinsic Rewards and Advantages
        ir_mean, ir_std = (np.mean(intrinsic_rewards), np.std(intrinsic_rewards))
        intrinsic_rewards = (intrinsic_rewards - ir_mean) / ir_std
        # int_deltas = intrinsic_rewards[:-1] + self.gamma * values[1:] - values[:-1]
        int_deltas = intrinsic_rewards[:-1] - intrinsic_rewards[1:]
        self.intrinsic_advantage_buffer[path_slice] = int_deltas

        self.intrinsic_reward_buffer[path_slice] = discounted_cumulative_sums(
            intrinsic_rewards, self.gamma
        )[:-1]

        # Overall Advantages

        self.total_advantage_buffer[path_slice] = \
            self.extrinsic_advantage_buffer[path_slice] + self.intrinsic_advantage_buffer[path_slice]

        # Overall Rewards

        self.total_reward_buffer[path_slice] = \
            self.extrinsic_reward_buffer[path_slice] + self.intrinsic_reward_buffer[path_slice]

        self.trajectory_start_index = self.pointer

    def get(self):
        # Get all data of the buffer, normalize the advantages and intrinsic rewards
        self.pointer, self.trajectory_start_index = 0, 0
        advantage_mean, advantage_std = (
            np.mean(self.total_advantage_buffer),
            np.std(self.total_advantage_buffer),
        )
        self.total_advantage_buffer = (self.total_advantage_buffer - advantage_mean) / advantage_std

        return (
            self.observation_buffer,
            self.action_buffer,
            self.total_advantage_buffer,
            self.intrinsic_reward_buffer,
            self.extrinsic_reward_buffer,
            self.logprobability_buffer
        )
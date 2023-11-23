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
        self.pointer, self.trajectory_start_index = 0, 0

        return (
            self.observation_buffer,
            self.action_buffer,
            self.advantage_buffer,
            self.return_buffer,
            self.logprobability_buffer,
        )


class RNDBuffer:

    def __init__(self, observation_dimensions, size, gamma=0.99, lam=0.95, intrinsic_weight=0.2):
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
        self.extrinsic_return_buffer = np.zeros(size, dtype=np.float32)
        self.intrinsic_return_buffer = np.zeros(size, dtype=np.float32)
        self.total_return_buffer = np.zeros(size, dtype=np.float32)
        self.value_buffer = np.zeros(size, dtype=np.float32)
        self.logprobability_buffer = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.intrinsic_weight = intrinsic_weight
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
        extrinsic_rewards = np.append(self.extrinsic_reward_buffer[path_slice], last_value)  # episodic
        intrinsic_rewards = np.append(self.intrinsic_reward_buffer[path_slice], 0)  # non-episodic
        values = np.append(self.value_buffer[path_slice], last_value)

        # Extrinsic Returns and Advantages
        ex_deltas = extrinsic_rewards[:-1] + self.gamma * values[1:] - values[:-1]
        self.extrinsic_advantage_buffer[path_slice] = discounted_cumulative_sums(
            ex_deltas, self.gamma * self.lam
        )
        self.extrinsic_return_buffer[path_slice] = discounted_cumulative_sums(
            extrinsic_rewards, self.gamma
        )[:-1]


        # Intrinsic Returns and Advantages
        int_deltas = intrinsic_rewards[:-1] + self.gamma * values[1:] - values[:-1]
        self.intrinsic_advantage_buffer[path_slice] = discounted_cumulative_sums(
            int_deltas, self.gamma * self.lam
        )
        self.intrinsic_return_buffer[path_slice] = discounted_cumulative_sums(
            intrinsic_rewards, self.gamma
        )[:-1]

        self.trajectory_start_index = self.pointer

    def get(self):

        # Get all data of the buffer, normalise int returns and advs, combine int and ext returns and advantages

        self.pointer, self.trajectory_start_index = 0, 0

        # Normalise intrinsic returns and advantages

        int_rew_mean, int_rew_std = (
                np.mean(self.intrinsic_return_buffer),
                np.std(self.intrinsic_return_buffer),
            )
        self.intrinsic_return_buffer = (self.intrinsic_return_buffer - int_rew_mean) / int_rew_std

        int_adv_mean, int_adv_std = (
            np.mean(self.intrinsic_advantage_buffer),
            np.std(self.intrinsic_advantage_buffer),
        )
        self.intrinsic_advantage_buffer = (self.intrinsic_advantage_buffer - int_adv_mean) / int_adv_std

        # Overall Advantages

        self.total_advantage_buffer = (self.extrinsic_advantage_buffer * (1 - self.intrinsic_weight)) + \
                                      (self.intrinsic_advantage_buffer * self.intrinsic_weight)

        # Overall Returns

        self.total_return_buffer = (self.extrinsic_return_buffer * (1 - self.intrinsic_weight)) + \
                                   (self.intrinsic_return_buffer * self.intrinsic_weight)

        return (
            self.observation_buffer,
            self.action_buffer,
            self.total_advantage_buffer,
            self.total_return_buffer,
            self.logprobability_buffer
        )
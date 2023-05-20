import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import gymnasium as gym
import scipy.signal

"""
## Functions and class
"""

def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Buffer:
    # Buffer for storing trajectories
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
        # Finish the trajectory by normalising intrinsic rewards, computing advantage estimates and rewards-to-go
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
        int_deltas = intrinsic_rewards[:-1] + self.gamma * values[1:] - values[:-1]
        self.intrinsic_advantage_buffer[path_slice] = discounted_cumulative_sums(
            int_deltas, self.gamma * self.lam
        )
        self.intrinsic_reward_buffer[path_slice] = discounted_cumulative_sums(
            extrinsic_rewards, self.gamma
        )[:-1]

        # 1. Should intrinsic rewards be discounted in the same way extrinsic rewards are?
        # 2. Should advantages be calculated for ex and int seperately, then added, or int rewards added to ex then advantages calculated.

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
            self.total_reward_buffer,
            self.logprobability_buffer
        )


def mlp(x, sizes, activation=tf.tanh, output_activation=None):
    # Build a feedforward neural network
    for size in sizes[:-1]:
        x = layers.Dense(units=size, activation=activation)(x)
    return layers.Dense(units=sizes[-1], activation=output_activation)(x)


def logprobabilities(logits, a):
    # Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the actor)
    logprobabilities_all = tf.nn.log_softmax(logits)
    logprobability = tf.reduce_sum(
        tf.one_hot(a, num_actions) * logprobabilities_all, axis=1
    )
    return logprobability


def calculate_intrinsic_reward(predictor_network, target_network, observation):
    prediction = predictor_network(observation)
    target = target_network(observation)
    error = np.square(target - prediction)
    return error


# Sample action from actor
@tf.function
def sample_action(observation):
    logits = actor(observation)
    action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
    return logits, action


# Train the policy by maximizing the PPO-Clip objective
@tf.function
def train_policy(
    observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
):

    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        ratio = tf.exp(
            logprobabilities(actor(observation_buffer), action_buffer)
            - logprobability_buffer
        )
        min_advantage = tf.where(
            advantage_buffer > 0,
            (1 + clip_ratio) * advantage_buffer,
            (1 - clip_ratio) * advantage_buffer,
        )

        policy_loss = -tf.reduce_mean(
            tf.minimum(ratio * advantage_buffer, min_advantage)
        )
    policy_grads = tape.gradient(policy_loss, actor.trainable_variables)
    policy_optimizer.apply_gradients(zip(policy_grads, actor.trainable_variables))



# Train the value function by regression on mean-squared error
@tf.function
def train_value_function(observation_buffer, return_buffer):
    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        value_loss = tf.reduce_mean((return_buffer - critic(observation_buffer)) ** 2)
    value_grads = tape.gradient(value_loss, critic.trainable_variables)
    value_optimizer.apply_gradients(zip(value_grads, critic.trainable_variables))

# Train the rnd predictor by regression on mean-squared error

@tf.function
def train_rnd_predictor(observation_buffer):
    with tf.GradientTape() as tape:
        rnd_loss = tf.reduce_mean((rnd_target(observation_buffer) - rnd_predictor(observation_buffer)) ** 2)
    rnd_grads = tape.gradient(rnd_loss, rnd_predictor.trainable_variables)
    rnd_predictor_optimiser.apply_gradients(zip(rnd_grads, rnd_predictor.trainable_variables))

"""
## Hyperparameters
"""

# Hyperparameters of the PPO algorithm
steps_per_epoch = 4000
epochs = 30
gamma = 0.99
clip_ratio = 0.2
policy_learning_rate = 3e-4
value_function_learning_rate = 1e-3
rnd_predictor_learning_rate = 1e-3
train_policy_iterations = 80
train_value_iterations = 80
train_rnd_iterations = 80
lam = 0.97
target_kl = 0.01
hidden_sizes = (64, 64)

# True if you want to render the environment

"""
## Initializations
"""

# Initialize the environment and get the dimensionality of the
# observation space and the number of possible actions
env = gym.make("CartPole-v1", render_mode=None)
observation_dimensions = env.observation_space.shape[0]
num_actions = env.action_space.n

# Initialize the buffer
buffer = Buffer(observation_dimensions, steps_per_epoch)

# Initialize the actor, critic and RND component as keras models
observation_input = keras.Input(shape=(observation_dimensions,), dtype=tf.float32)
logits = mlp(observation_input, list(hidden_sizes) + [num_actions], tf.tanh, None)
actor = keras.Model(inputs=observation_input, outputs=logits)
value = tf.squeeze(
    mlp(observation_input, list(hidden_sizes) + [1], tf.tanh, None), axis=1
)
critic = keras.Model(inputs=observation_input, outputs=value)
rnd_predictor_logits = tf.squeeze(mlp(observation_input, list(hidden_sizes) + [1], keras.activations.relu, None), axis=1)
rnd_target_logits = tf.squeeze(mlp(observation_input, list(hidden_sizes) + [1], keras.activations.relu, None), axis=1)
rnd_predictor = keras.Model(inputs=observation_input, outputs=rnd_predictor_logits)
rnd_target = keras.Model(inputs=observation_input, outputs=rnd_target_logits)


# Initialize the policy and the value function optimizers
policy_optimizer = keras.optimizers.Adam(learning_rate=policy_learning_rate)
value_optimizer = keras.optimizers.Adam(learning_rate=value_function_learning_rate)
rnd_predictor_optimiser = keras.optimizers.Adam(learning_rate=rnd_predictor_learning_rate)

# Initialize the observation, episode return and episode length
observation, episode_return, episode_length = env.reset()[0], 0, 0

"""
## Train
"""
# Iterate over the number of epochs
for epoch in range(epochs):
    # Initialize the sum of the returns, lengths and number of episodes for each epoch
    sum_return = 0
    sum_length = 0
    num_episodes = 0
    #if epoch in [0, 14, 29]:
        #env.render()

    # Iterate over the steps of each epoch
    for t in range(steps_per_epoch):

        # Get the logits, action, and take one step in the environment
        if len(observation) == 2:
            observation = observation[0]
        observation = np.reshape(observation, [1, env.observation_space.shape[0]])
        logits, action = sample_action(observation)
        observation_new, extrinsic_reward, done, _, _ = env.step(action[0].numpy())
        observation_new = np.reshape(observation_new, [1, env.observation_space.shape[0]])
        intrinsic_reward = calculate_intrinsic_reward(rnd_predictor, rnd_target, observation_new)
        episode_return += extrinsic_reward
        episode_length += 1

        # Get the value and log-probability of the action
        value_t = critic(observation)
        logprobability_t = logprobabilities(logits, action)

        # Store obs, act, rew, v_t, logp_pi_t
        buffer.store(observation, action, extrinsic_reward, intrinsic_reward, value_t, logprobability_t)

        # Update the observation
        observation = observation_new

        # Finish trajectory if reached to a terminal state
        terminal = done
        if terminal or (t == steps_per_epoch - 1):
            last_value = 0 if done else critic(observation.reshape(1, -1))
            buffer.finish_trajectory(last_value)
            sum_return += episode_return
            sum_length += episode_length
            num_episodes += 1
            observation, episode_return, episode_length = env.reset(), 0, 0

    # Get values from the buffer
    (
        observation_buffer,
        action_buffer,
        total_advantage_buffer,
        intrinsic_reward_buffer,
        total_reward_buffer,
        logprobability_buffer,
    ) = buffer.get()

    # Update the policy
    for _ in range(train_policy_iterations):
        train_policy(observation_buffer, action_buffer, logprobability_buffer, total_advantage_buffer)

    # Update the value function
    for _ in range(train_value_iterations):
        train_value_function(observation_buffer, total_reward_buffer)

    for _ in range(train_rnd_iterations):
        train_rnd_predictor(observation_buffer)

    # Print mean return and length for each epoch
    print(
        f" Epoch: {epoch + 1}. Mean Return: {sum_return / num_episodes}. Mean Length: {sum_length / num_episodes}"
    )

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import gymnasium as gym
from buffer import RNDBuffer as Buffer


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

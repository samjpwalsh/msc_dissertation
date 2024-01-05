import numpy as np
import random
import os
import tensorflow as tf
from tensorflow import keras
from keras.models import clone_model
from dissertation_files.agents.buffer import DQNBuffer, PPOBuffer, RNDBuffer
from dissertation_files.agents.utils import mlp, logprobabilities


class RandomAgent:
    def __init__(self, action_dimensions):
        self.action_dimensions = action_dimensions

    def sample_action(self):
        return random.randrange(self.action_dimensions)


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
        self.buffer = DQNBuffer(observation_dimensions, memory_size, batch_size)
        self.model, self.optimizer = self.build_model(hidden_sizes, input_activation, output_activation, learning_rate)
        self.target_model = self.build_target_model()

    def build_model(self, hidden_sizes, input_activation, output_activation, learning_rate):
        observation_input = keras.Input(shape=(self.observation_dimensions,), dtype=tf.float32)
        logits = mlp(observation_input, list(hidden_sizes) + [self.action_dimensions], input_activation, output_activation)
        model = keras.Model(inputs=observation_input, outputs=logits)
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
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
            loss = tf.losses.huber(targets, q_values_actions)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.min_epsilon, self.epsilon)

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())


class PPOAgent:
    def __init__(self, observation_dimensions, action_dimensions, memory_size, hidden_sizes, input_activation,
                 output_activation, actor_learning_rate, critic_learning_rate, clip_ratio, gamma, lam):

        self.observation_dimensions = observation_dimensions
        self.action_dimensions = action_dimensions
        self.memory_size = memory_size
        self.clip_ratio = clip_ratio
        self.buffer = PPOBuffer(observation_dimensions, memory_size, gamma, lam)
        self.actor, self.actor_optimiser = self.build_actor(
            hidden_sizes,
            input_activation,
            output_activation,
            actor_learning_rate
        )
        self.critic, self.critic_optimiser = self.build_critic(
            hidden_sizes,
            input_activation,
            output_activation,
            critic_learning_rate
        )

    def build_actor(self, hidden_sizes, input_activation, output_activation, learning_rate):
        observation_input = keras.Input(shape=(self.observation_dimensions,), dtype=tf.float32)
        logits = mlp(observation_input, list(hidden_sizes) + [self.action_dimensions], input_activation, output_activation)
        actor = keras.Model(inputs=observation_input, outputs=logits)
        actor_optimiser = keras.optimizers.Adam(learning_rate=learning_rate)

        return actor, actor_optimiser

    def build_critic(self, hidden_sizes, input_activation, output_activation, learning_rate):
        observation_input = keras.Input(shape=(self.observation_dimensions,), dtype=tf.float32)
        logits = tf.squeeze(
                    mlp(observation_input, list(hidden_sizes) + [1], input_activation, output_activation), axis=1
                )
        critic = keras.Model(inputs=observation_input, outputs=logits)
        critic_optimiser = keras.optimizers.Adam(learning_rate=learning_rate)

        return critic, critic_optimiser

    @tf.function
    def sample_action(self, observation):
        logits = self.actor(observation)
        action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)

        return logits, action

    @tf.function
    def train_actor(
        self, observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
    ):

        with tf.GradientTape() as tape:
            ratio = tf.exp(
                logprobabilities(self.actor(observation_buffer), action_buffer, self.action_dimensions)
                - logprobability_buffer
            )
            min_advantage = tf.where(
                advantage_buffer > 0,
                (1 + self.clip_ratio) * advantage_buffer,
                (1 - self.clip_ratio) * advantage_buffer,
            )

            actor_loss = -tf.reduce_mean(
                tf.minimum(ratio * advantage_buffer, min_advantage)
            )
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimiser.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

    @tf.function
    def train_critic(self, observation_buffer, return_buffer):
        with tf.GradientTape() as tape:
            critic_loss = tf.reduce_mean((return_buffer - self.critic(observation_buffer)) ** 2)
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimiser.apply_gradients(zip(critic_grads, self.critic.trainable_variables))


class RNDAgent:
    def __init__(self, observation_dimensions, action_dimensions, memory_size, hidden_sizes, input_activation,
                 output_activation, actor_learning_rate, critic_learning_rate, rnd_predictor_learning_rate,
                 clip_ratio, gamma, lam, intrinsic_weight):

        self.observation_dimensions = observation_dimensions
        self.action_dimensions = action_dimensions
        self.memory_size = memory_size
        self.clip_ratio = clip_ratio
        self.buffer = RNDBuffer(observation_dimensions, memory_size, gamma, lam, intrinsic_weight)
        self.actor, self.actor_optimiser = self.build_actor(
            hidden_sizes,
            input_activation,
            output_activation,
            actor_learning_rate
        )
        self.critic, self.critic_optimiser = self.build_critic(
            hidden_sizes,
            input_activation,
            output_activation,
            critic_learning_rate
        )
        self.rnd_predictor, self.rnd_target, self.rnd_predictor_optimiser = self.build_rnd_predictor_and_target(
            hidden_sizes,
            input_activation,
            output_activation,
            rnd_predictor_learning_rate
        )

    def build_actor(self, hidden_sizes, input_activation, output_activation, learning_rate):
        observation_input = keras.Input(shape=(self.observation_dimensions,), dtype=tf.float32)
        logits = mlp(observation_input, list(hidden_sizes) + [self.action_dimensions], input_activation, output_activation)
        actor = keras.Model(inputs=observation_input, outputs=logits)
        actor_optimiser = keras.optimizers.Adam(learning_rate=learning_rate)

        return actor, actor_optimiser

    def build_critic(self, hidden_sizes, input_activation, output_activation, learning_rate):
        observation_input = keras.Input(shape=(self.observation_dimensions,), dtype=tf.float32)
        logits = tf.squeeze(
                    mlp(observation_input, list(hidden_sizes) + [1], input_activation, output_activation), axis=1
                )
        critic = keras.Model(inputs=observation_input, outputs=logits)
        critic_optimiser = keras.optimizers.Adam(learning_rate=learning_rate)

        return critic, critic_optimiser

    def build_rnd_predictor_and_target(self, hidden_sizes, input_activation, output_activation, learning_rate):
        observation_input = keras.Input(shape=(self.observation_dimensions,), dtype=tf.float32)
        predictor_logits = tf.squeeze(
            mlp(observation_input, list(hidden_sizes) + [1], input_activation, output_activation), axis=1
        )
        target_logits = tf.squeeze(
            mlp(observation_input, list(hidden_sizes) + [1], input_activation, output_activation), axis=1
        )
        rnd_predictor = keras.Model(inputs=observation_input, outputs=predictor_logits)
        rnd_target = keras.Model(inputs=observation_input, outputs=target_logits)
        rnd_predictor_optimiser = keras.optimizers.Adam(learning_rate=learning_rate)

        return rnd_predictor, rnd_target, rnd_predictor_optimiser

    @tf.function
    def sample_action(self, observation):
        logits = self.actor(observation)
        action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)

        return logits, action

    def calculate_intrinsic_reward(self, observation):
        prediction = self.rnd_predictor(observation)
        target = self.rnd_target(observation)
        error = float(np.square(target - prediction))
        return error

    @tf.function
    def train_actor(
        self, observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
    ):

        with tf.GradientTape() as tape:
            ratio = tf.exp(
                logprobabilities(self.actor(observation_buffer), action_buffer, self.action_dimensions)
                - logprobability_buffer
            )
            min_advantage = tf.where(
                advantage_buffer > 0,
                (1 + self.clip_ratio) * advantage_buffer,
                (1 - self.clip_ratio) * advantage_buffer,
            )

            actor_loss = -tf.reduce_mean(
                tf.minimum(ratio * advantage_buffer, min_advantage)
            )
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimiser.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

    @tf.function
    def train_critic(self, observation_buffer, return_buffer):
        with tf.GradientTape() as tape:
            critic_loss = tf.reduce_mean((return_buffer - self.critic(observation_buffer)) ** 2)
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimiser.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

    @tf.function
    def train_rnd_predictor(self, observation_buffer):
        with tf.GradientTape() as tape:
            rnd_loss = tf.reduce_mean((self.rnd_target(observation_buffer) - self.rnd_predictor(observation_buffer)) ** 2)
        rnd_grads = tape.gradient(rnd_loss, self.rnd_predictor.trainable_variables)
        self.rnd_predictor_optimiser.apply_gradients(zip(rnd_grads, self.rnd_predictor.trainable_variables))

    def save_models(self, filepath):
        actor_model_prefix = os.path.join(filepath, "actor_model/ckpt")
        actor_optimiser_prefix = os.path.join(filepath, "actor_optimiser/ckpt")
        critic_model_prefix = os.path.join(filepath, "critic_model/ckpt")
        critic_optimiser_prefix = os.path.join(filepath, "actor_optimiser/ckpt")
        rnd_predictor_model_prefix = os.path.join(filepath, "rnd_predictor_model/ckpt")
        rnd_predictor_optimiser_prefix = os.path.join(filepath, "rnd_predictor_optimiser/ckpt")
        rnd_target_model_prefix = os.path.join(filepath, "rnd_target_model/ckpt")

        actor_model_checkpoint = tf.train.Checkpoint(model=self.actor)
        actor_optimiser_checkpoint = tf.train.Checkpoint(optimiser=self.actor_optimiser)
        critic_model_checkpoint = tf.train.Checkpoint(model=self.critic)
        critic_optimiser_checkpoint = tf.train.Checkpoint(optimiser=self.critic_optimiser)
        rnd_predictor_model_checkpoint = tf.train.Checkpoint(model=self.rnd_predictor)
        rnd_predictor_optimiser_checkpoint = tf.train.Checkpoint(optimiser=self.rnd_predictor_optimiser)
        rnd_target_model_checkpoint = tf.train.Checkpoint(model=self.rnd_target)

        actor_model_checkpoint.save(file_prefix=actor_model_prefix)
        actor_optimiser_checkpoint.save(file_prefix=actor_optimiser_prefix)
        critic_model_checkpoint.save(file_prefix=critic_model_prefix)
        critic_optimiser_checkpoint.save(file_prefix=critic_optimiser_prefix)
        rnd_predictor_model_checkpoint.save(file_prefix=rnd_predictor_model_prefix)
        rnd_predictor_optimiser_checkpoint.save(file_prefix=rnd_predictor_optimiser_prefix)
        rnd_target_model_checkpoint.save(file_prefix=rnd_target_model_prefix)


class PretrainedRNDAgent(RNDAgent):

    def __init__(self, observation_dimensions, action_dimensions, memory_size, hidden_sizes, input_activation,
                 output_activation, actor_learning_rate, critic_learning_rate, rnd_predictor_learning_rate, clip_ratio,
                 gamma, lam, intrinsic_weight, checkpoint_filepath, restore_rnd_networks=True):

        super().__init__(observation_dimensions, action_dimensions, memory_size, hidden_sizes, input_activation,
                         output_activation, actor_learning_rate, critic_learning_rate, rnd_predictor_learning_rate,
                         clip_ratio, gamma, lam, intrinsic_weight)

        actor_model_prefix = os.path.join(checkpoint_filepath, "actor_model/ckpt-1")
        critic_model_prefix = os.path.join(checkpoint_filepath, "critic_model/ckpt-1")

        actor_model_checkpoint = tf.train.Checkpoint(model=self.actor)
        critic_model_checkpoint = tf.train.Checkpoint(model=self.critic)

        actor_model_checkpoint.restore(actor_model_prefix)
        critic_model_checkpoint.restore(critic_model_prefix)

        if restore_rnd_networks:
            rnd_predictor_model_prefix = os.path.join(checkpoint_filepath, "rnd_predictor_model/ckpt-1")
            rnd_predictor_optimiser_prefix = os.path.join(checkpoint_filepath, "rnd_predictor_optimiser/ckpt-1")
            rnd_target_model_prefix = os.path.join(checkpoint_filepath, "rnd_target_model/ckpt-1")

            rnd_predictor_model_checkpoint = tf.train.Checkpoint(model=self.rnd_predictor)
            rnd_predictor_optimiser_checkpoint = tf.train.Checkpoint(optimiser=self.rnd_predictor_optimiser)
            rnd_target_model_checkpoint = tf.train.Checkpoint(model=self.rnd_target)

            rnd_predictor_model_checkpoint.restore(rnd_predictor_model_prefix)
            rnd_predictor_optimiser_checkpoint.restore(rnd_predictor_optimiser_prefix)
            rnd_target_model_checkpoint.restore(rnd_target_model_prefix)

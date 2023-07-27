import numpy as np
from dissertation_files.agents.utils import logprobabilities


def dqn_training_loop(episodes, agent, env, observation_dimensions, steps_target_model_update):

    reward_list = []
    step_counter = 0

    for episode in range(episodes):
        observation = env.reset()[0]
        observation = np.reshape(observation, [1, observation_dimensions])
        done = False
        episode_reward = 0
        while not done:
            action = agent.sample_action(observation)
            next_observation, reward, done, truncated, _ = env.step(action)
            if truncated:
                done = True
            next_observation = np.reshape(next_observation, [1, observation_dimensions])
            agent.buffer.store(observation, action, next_observation, reward, done)
            agent.train_model()
            if step_counter % steps_target_model_update == 0 and step_counter != 0:
                agent.update_target_model()
            episode_reward += reward
            observation = next_observation
            step_counter += 1
        print(f"episode: {episode + 1}/{episodes}, score: {episode_reward}")
        reward_list.append(episode_reward)

    return reward_list


def ppo_training_loop(epochs, agent, env, observation_dimensions, action_dimensions, steps_per_epoch,
                      train_actor_iterations, train_critic_iterations):

    observation, episode_reward = env.reset()[0], 0
    reward_list = []
    episodes = 0

    for epoch in range(epochs):

        for t in range(steps_per_epoch):

            observation = np.reshape(observation, [1, observation_dimensions])
            logits, action = agent.sample_action(observation)
            observation_new, reward, done, truncated, _ = env.step(action[0].numpy())
            if truncated:
                done = True
            episode_reward += reward

            value_t = agent.critic(observation)
            logprobability_t = logprobabilities(logits, action, action_dimensions)

            agent.buffer.store(observation, action, reward, value_t, logprobability_t)

            observation = observation_new

            if done or (t == steps_per_epoch - 1):
                last_value = 0 if done else agent.critic(observation.reshape(1, -1))
                agent.buffer.finish_trajectory(last_value)
                print(f"episode: {episodes+1}, score: {episode_reward}")
                episodes += 1
                reward_list.append(episode_reward)
                observation, episode_reward = env.reset()[0], 0

        (
            observation_buffer,
            action_buffer,
            advantage_buffer,
            return_buffer,
            logprobability_buffer,
        ) = agent.buffer.get()

        for _ in range(train_actor_iterations):
            agent.train_actor(observation_buffer, action_buffer, logprobability_buffer, advantage_buffer)

        for _ in range(train_critic_iterations):
            agent.train_critic(observation_buffer, return_buffer)

    return episodes, reward_list


def rnd_training_loop(epochs, agent, env, observation_dimensions, action_dimensions, steps_per_epoch,
                      train_actor_iterations, train_critic_iterations, train_rnd_iterations):

    observation, episode_reward, episode_intrinsic_reward = env.reset()[0], 0, 0
    observation = np.reshape(observation, [1, observation_dimensions])
    reward_list = []
    intrinsic_reward_list = []
    episodes = 0

    for epoch in range(epochs):

        for t in range(steps_per_epoch):

            logits, action = agent.sample_action(observation)
            observation_new, extrinsic_reward, done, truncated, _ = env.step(action[0].numpy())
            if truncated:
                done = True
            observation_new = np.reshape(observation_new, [1, observation_dimensions])
            intrinsic_reward = agent.calculate_intrinsic_reward(observation_new)
            episode_reward += extrinsic_reward
            episode_intrinsic_reward += intrinsic_reward

            value_t = agent.critic(observation)
            logprobability_t = logprobabilities(logits, action, action_dimensions)

            agent.buffer.store(observation, action, extrinsic_reward, intrinsic_reward, value_t, logprobability_t)

            observation = observation_new

            if done or (t == steps_per_epoch - 1):
                last_value = 0 if done else agent.critic(observation.reshape(1, -1))
                agent.buffer.finish_trajectory(last_value)
                print(f"episode: {episodes + 1}, score: {episode_reward}, intrinsic reward: {episode_intrinsic_reward}")
                episodes += 1
                reward_list.append(episode_reward)
                intrinsic_reward_list.append(episode_intrinsic_reward)
                observation, episode_reward, episode_intrinsic_reward = env.reset()[0], 0, 0
                observation = np.reshape(observation, [1, observation_dimensions])

        (
            observation_buffer,
            action_buffer,
            total_advantage_buffer,
            intrinsic_reward_buffer,
            total_reward_buffer,
            logprobability_buffer,
        ) = agent.buffer.get()

        for _ in range(train_actor_iterations):
            agent.train_actor(observation_buffer, action_buffer, logprobability_buffer, total_advantage_buffer)

        for _ in range(train_critic_iterations):
            agent.train_critic(observation_buffer, total_reward_buffer)
            # should the critic be trained on the total reward in RND or just extrinsic/intrinsic

        for _ in range(train_rnd_iterations):
            agent.train_rnd_predictor(observation_buffer)

    return episodes, reward_list, intrinsic_reward_list

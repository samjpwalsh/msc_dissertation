import numpy as np
from dissertation_files.agents.utils import logprobabilities
from dissertation_files.agents.evaluation import run_random_evaluation, run_dqn_evaluation, run_ppo_evaluation, \
    run_rnd_evaluation


def dqn_training_loop(epochs, agent, env, observation_dimensions, steps_per_epoch, steps_target_model_update,
                      video_folder, eval_env=None, eval_epoch_frequency=1, eval_episodes_per_epoch=10):

    observation = env.reset()[0]
    average_reward_list = []
    eval_average_reward_list = []
    first_time_visits = {env.agent_pos: 0}
    step_counter = 0
    episode_reward = 0

    if eval_env is not None:
        eval_ave_reward = run_dqn_evaluation(agent, eval_env, eval_episodes_per_epoch, video_folder, 0)
        eval_average_reward_list.append(eval_ave_reward)

    for epoch in range(epochs):

        epoch_total_reward = 0
        epoch_episodes = 0

        if epoch == 0:
            full_episode = True
        else:
            full_episode = False

        for t in range(steps_per_epoch):

            observation = np.reshape(observation, [1, observation_dimensions])
            action = agent.sample_action(observation)
            observation_new, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            if truncated:
                done = True
            observation_new = np.reshape(observation_new, [1, observation_dimensions])
            agent.buffer.store(observation, action, observation_new, reward, done)
            agent.train_model()
            if step_counter % steps_target_model_update == 0 and step_counter != 0:
                agent.update_target_model()
            observation = observation_new
            step_counter += 1

            try:
                if env.agent_pos not in first_time_visits:
                    first_time_visits[env.agent_pos] = step_counter
            except AttributeError:
                first_time_visits = None

            if done:
                if full_episode:
                    epoch_total_reward += episode_reward
                    epoch_episodes += 1
                else:
                    full_episode = True
                episode_reward = 0
                observation = env.reset()[0]

        average_score_per_episode = epoch_total_reward / epoch_episodes
        print(f"Epoch: {epoch + 1}/{epochs}, Average score per episode: {average_score_per_episode}")
        average_reward_list.append(average_score_per_episode)

        if eval_env is not None and ((epoch + 1) % eval_epoch_frequency == 0):
            eval_ave_reward = run_dqn_evaluation(agent, eval_env, eval_episodes_per_epoch, video_folder, epoch+1)
            eval_average_reward_list.append(eval_ave_reward)

    if eval_env is None:
        return average_reward_list, first_time_visits
    else:
        return eval_average_reward_list, first_time_visits


def ppo_training_loop(epochs, agent, env, observation_dimensions, action_dimensions, steps_per_epoch,
                      train_actor_iterations, train_critic_iterations, video_folder, eval_env=None,
                      eval_epoch_frequency=1, eval_episodes_per_epoch=10):

    observation = env.reset()[0]
    average_reward_list = []
    eval_average_reward_list = []
    first_time_visits = {env.agent_pos: 0}
    episode_reward = 0
    step_counter = 0

    if eval_env is not None:
        eval_ave_reward = run_ppo_evaluation(agent, eval_env, eval_episodes_per_epoch, video_folder, 0)
        eval_average_reward_list.append(eval_ave_reward)

    for epoch in range(epochs):

        epoch_total_reward = 0
        epoch_episodes = 0

        if epoch == 0:
            full_episode = True
        else:
            full_episode = False

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

            step_counter += 1
            try:
                if env.agent_pos not in first_time_visits:
                    first_time_visits[env.agent_pos] = step_counter
            except AttributeError:
                first_time_visits = None

            if done or (t == steps_per_epoch - 1):
                last_value = 0 if done else agent.critic(observation.reshape(1, -1))
                agent.buffer.finish_trajectory(last_value)

            if done:
                if full_episode:
                    epoch_total_reward += episode_reward
                    epoch_episodes += 1
                else:
                    full_episode = True
                episode_reward = 0
                observation = env.reset()[0]

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

        average_score_per_episode = epoch_total_reward / epoch_episodes
        print(f"Epoch: {epoch + 1}/{epochs}, Average score per episode: {average_score_per_episode}")
        average_reward_list.append(average_score_per_episode)

        if eval_env is not None and ((epoch + 1) % eval_epoch_frequency == 0):
            eval_ave_reward = run_ppo_evaluation(agent, eval_env, eval_episodes_per_epoch, video_folder, epoch+1)
            eval_average_reward_list.append(eval_ave_reward)

    if eval_env is None:
        return average_reward_list, first_time_visits
    else:
        return eval_average_reward_list, first_time_visits


def rnd_training_loop(epochs, agent, env, observation_dimensions, action_dimensions, steps_per_epoch,
                      train_actor_iterations, train_critic_iterations, train_rnd_iterations, video_folder,
                      eval_env=None, eval_epoch_frequency=1, eval_episodes_per_epoch=10):

    observation = env.reset()[0]
    observation = np.reshape(observation, [1, observation_dimensions])
    average_intrinsic_reward_list = []
    average_reward_list = []
    eval_average_reward_list = []
    eval_average_intrinsic_reward_list = []
    episode_reward = 0
    episode_intrinsic_reward = 0
    step_counter = 0
    first_time_visits = {env.agent_pos: 0}

    if eval_env is not None:
        eval_ave_reward, eval_ave_intrinsic_reward = run_rnd_evaluation(agent, eval_env, eval_episodes_per_epoch,
                                                                        video_folder, 0)
        eval_average_reward_list.append(eval_ave_reward)
        eval_average_intrinsic_reward_list.append(eval_ave_intrinsic_reward)

    for epoch in range(epochs):

        epoch_total_reward = 0
        epoch_total_intrinsic_reward = 0
        epoch_episodes = 0

        if epoch == 0:
            full_episode = True
        else:
            full_episode = False

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

            step_counter += 1
            try:
                if env.agent_pos not in first_time_visits:
                    first_time_visits[env.agent_pos] = step_counter
            except AttributeError:
                first_time_visits = None

            if done or (t == steps_per_epoch - 1):
                last_value = 0 if done else agent.critic(observation.reshape(1, -1))
                agent.buffer.finish_trajectory(last_value)

            if done:
                if full_episode:
                    epoch_total_reward += episode_reward
                    epoch_total_intrinsic_reward += episode_intrinsic_reward
                    epoch_episodes += 1
                else:
                    full_episode = True
                episode_reward = 0
                episode_intrinsic_reward = 0
                observation = env.reset()[0]
                observation = np.reshape(observation, [1, observation_dimensions])
        (
            observation_buffer,
            action_buffer,
            total_advantage_buffer,
            intrinsic_reward_buffer,
            extrinsic_reward_buffer,
            logprobability_buffer,
        ) = agent.buffer.get()

        for _ in range(train_actor_iterations):
            agent.train_actor(observation_buffer, action_buffer, logprobability_buffer, total_advantage_buffer)

        for _ in range(train_critic_iterations):
            agent.train_critic(observation_buffer, extrinsic_reward_buffer)

        for _ in range(train_rnd_iterations):
            agent.train_rnd_predictor(observation_buffer)

        average_score_per_episode = epoch_total_reward / epoch_episodes
        average_intrinsic_reward_per_episode = epoch_total_intrinsic_reward / epoch_episodes
        print(f"Epoch: {epoch + 1}/{epochs}, Average score per episode: {average_score_per_episode}, "
              f"Average intrinsic per episode: {average_intrinsic_reward_per_episode}")
        average_reward_list.append(average_score_per_episode)
        average_intrinsic_reward_list.append(average_intrinsic_reward_per_episode)

        if eval_env is not None and ((epoch + 1) % eval_epoch_frequency == 0):
            eval_ave_reward, eval_ave_intrinsic_reward = run_rnd_evaluation(agent, eval_env, eval_episodes_per_epoch,
                                                                            video_folder, epoch+1)
            eval_average_reward_list.append(eval_ave_reward)
            eval_average_intrinsic_reward_list.append(eval_ave_intrinsic_reward)

    if eval_env is None:
        return average_reward_list, average_intrinsic_reward_list, first_time_visits
    else:
        return eval_average_reward_list, eval_average_intrinsic_reward_list, first_time_visits


def random_play_loop(epochs, agent, env, steps_per_epoch, video_folder, eval_env=None, eval_epoch_frequency=1,
                     eval_episodes_per_epoch=10):

    env.reset()
    average_reward_list = []
    eval_average_reward_list = []
    first_time_visits = {env.agent_pos: 0}
    step_counter = 0
    episode_reward = 0

    if eval_env is not None:
        eval_ave_reward = run_random_evaluation(agent, eval_env, eval_episodes_per_epoch, video_folder, 0)
        eval_average_reward_list.append(eval_ave_reward)

    for epoch in range(epochs):

        epoch_total_reward = 0
        epoch_episodes = 0

        if epoch == 0:
            full_episode = True
        else:
            full_episode = False

        for t in range(steps_per_epoch):

            action = agent.sample_action()
            _, reward, done, truncated, _ = env.step(action)
            if truncated:
                done = True
            episode_reward += reward

            step_counter += 1
            try:
                if env.agent_pos not in first_time_visits:
                    first_time_visits[env.agent_pos] = step_counter
            except AttributeError:
                first_time_visits = None

            if done:
                if full_episode:
                    epoch_total_reward += episode_reward
                    epoch_episodes += 1
                else:
                    full_episode = True
                episode_reward = 0
                env.reset()

        if epoch_episodes == 0:
            x=1
        average_score_per_episode = epoch_total_reward / epoch_episodes
        print(f"Epoch: {epoch + 1}/{epochs}, Average score per episode: {average_score_per_episode}")
        average_reward_list.append(average_score_per_episode)

        if eval_env is not None and ((epoch + 1) % eval_epoch_frequency == 0):
            eval_ave_reward = run_random_evaluation(agent, eval_env, eval_episodes_per_epoch, video_folder, epoch+1)
            eval_average_reward_list.append(eval_ave_reward)

    if eval_env is None:
        return average_reward_list, first_time_visits
    else:
        return eval_average_reward_list, first_time_visits

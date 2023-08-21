import numpy as np
from matplotlib import pyplot as plt


def run_dqn_evaluation(agent, eval_env, episodes):
    observation_dimensions = len(eval_env.reset()[0])
    total_reward = 0
    for episode in range(episodes):
        observation = eval_env.reset()[0]
        done = False
        while not done:
            eval_env.render()
            observation = np.reshape(observation, [1, observation_dimensions])
            action = agent.sample_action(observation)
            observation_new, reward, done, truncated, _ = eval_env.step(action)
            total_reward += reward
            if truncated:
                done = True
            observation_new = np.reshape(observation_new, [1, observation_dimensions])
            observation = observation_new
    average_reward = total_reward / episodes

    return average_reward


def run_ppo_evaluation(agent, eval_env, episodes):
    observation_dimensions = len(eval_env.reset()[0])
    total_reward = 0
    for episode in range(episodes):
        observation = eval_env.reset()[0]
        done = False
        while not done:
            eval_env.render()
            observation = np.reshape(observation, [1, observation_dimensions])
            logits, action = agent.sample_action(observation)
            observation_new, reward, done, truncated, _ = eval_env.step(action[0].numpy())
            total_reward += reward
            if truncated:
                done = True
            observation = observation_new
    average_reward = total_reward / episodes

    return average_reward


def run_rnd_evaluation(agent, eval_env, episodes):
    observation_dimensions = len(eval_env.reset()[0])
    total_reward = 0
    total_intrinsic_reward = 0

    for episode in range(episodes):
        observation = eval_env.reset()[0]
        observation = np.reshape(observation, [1, observation_dimensions])
        done = False
        while not done:
            eval_env.render()
            logits, action = agent.sample_action(observation)
            observation_new, extrinsic_reward, done, truncated, _ = eval_env.step(action[0].numpy())
            total_reward += extrinsic_reward
            if truncated:
                done = True
            observation_new = np.reshape(observation_new, [1, observation_dimensions])
            intrinsic_reward = agent.calculate_intrinsic_reward(observation_new)
            total_intrinsic_reward += intrinsic_reward
            observation = observation_new
    average_reward = total_reward / episodes
    average_intrinsic_reward = total_intrinsic_reward / episodes

    return average_reward, average_intrinsic_reward


def plot_evaluation_data(rewards, epochs, eval_frequency, steps_per_epoch):
    step_list = [(i * steps_per_epoch * eval_frequency) for i in range((epochs // eval_frequency) + 1)]
    if len(rewards) == 1:
        rewards = rewards[0]
        plt.plot(step_list, rewards)
        plt.ylabel("Average Extrinsic Reward")
        plt.xlabel("Training Steps")
    else:
        ext_rewards, int_rewards = rewards
        fig, ax1 = plt.subplots()
        ax1.plot(step_list, ext_rewards, label='Extrinsic Rewards')
        ax1.set_ylabel("Average Extrinsic Reward")
        ax1.set_xlabel("Training Steps")
        ax2 = ax1.twinx()
        ax2.plot(step_list, int_rewards, label='Intrinsic Rewards', color='red')
        ax2.set_ylabel("Average Intrinsic Reward")
        lines = ax1.get_lines() + ax2.get_lines()
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc='upper center')
    plt.xticks(step_list)
    plt.title("Average reward after training for x steps")
    plt.show()

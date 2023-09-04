import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from dissertation_files.environments.simple_env import SimpleEnv
from dissertation_files.environments.test_wall_env import TestWall
from dissertation_files.environments.minigrid_wrappers import FlatObsWrapper

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

def get_grid_representation(gridworld_env):
    full_grid = gridworld_env.grid.encode()
    simple_grid = []
    for row in full_grid:
        simple_row = []
        for cell in row:
            simple_row.append(cell[0])
        simple_grid.append(simple_row)
    return np.array(simple_grid).transpose()

def get_unvisitable_cells(grid):
    unvisitable_cells = {}
    grid_height = len(grid)
    grid_width = len(grid[0])
    for j in range(grid_height):
        for i in range(grid_width):
            if grid[i, j] == 2:  # wall
                unvisitable_cells[(j, i)] = -1
    return unvisitable_cells

def plot_exploration_heatmap(gridworld_env, first_time_visits):
    grid = get_grid_representation(gridworld_env)
    unvisitable_cells = get_unvisitable_cells(grid)
    updated_ftvs = first_time_visits.copy()
    updated_ftvs.update(unvisitable_cells)
    ser = pd.Series(list(updated_ftvs.values()),
                    index=pd.MultiIndex.from_tuples(updated_ftvs.keys()))
    df = ser.unstack().transpose().fillna(max(updated_ftvs.values()) + 1)
    cmap = sns.color_palette("coolwarm", as_cmap=True)
    cmap.set_under("grey")  # walls
    cmap.set_over("black")  # unvisited states
    sns.heatmap(df, vmin=0, vmax=max(updated_ftvs.values()), cmap=cmap, xticklabels=False, yticklabels=False)
    plt.show()

def plot_state_visit_percentage(gridworld_env, first_time_visits, epochs, steps_per_epoch):
    grid = get_grid_representation(gridworld_env)
    total_states = grid.shape[0] * grid.shape[1]
    unvisitable_cells = get_unvisitable_cells(grid)
    total_states -= len(unvisitable_cells)
    state_visit_steps = sorted(list(first_time_visits.values()))
    y = [((i+1)/total_states) * 100 for i in range(len(state_visit_steps))]
    y.append(y[-1])
    state_visit_steps.append(steps_per_epoch * epochs)
    plt.plot(state_visit_steps, y)
    plt.ylabel("Percentage of grid cells explored")
    plt.xlabel("Training Steps")
    plt.xlim(0, steps_per_epoch * epochs)
    plt.ylim(0, 100)
    plt.title("Percentage of the environment explored throughout training")
    plt.show()

import pickle
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# from gymnasium.utils.save_video import save_video
from dissertation_files.agents.utils import save_video, confidence_interval


def run_random_evaluation(agent, eval_env, episodes, video_folder, current_epoch):
    eval_env.reset()
    total_reward = 0
    frames = []
    for episode in range(episodes):
        eval_env.reset()
        done = False
        while not done:
            if episode == 0 and video_folder is not None:
                frames.append(eval_env.render())
            action = agent.sample_action()
            _, reward, done, truncated, _ = eval_env.step(action)
            total_reward += reward
            if truncated:
                done = True
        if episode == 0 and video_folder is not None:
            save_video(frames, video_folder, fps=eval_env.metadata["render_fps"],
                       name_prefix=f"random_epoch_{current_epoch}")

    average_reward = total_reward / episodes

    return average_reward


def run_dqn_evaluation(agent, eval_env, episodes, video_folder, current_epoch):
    observation_dimensions = len(eval_env.reset()[0])
    total_reward = 0
    frames = []
    for episode in range(episodes):
        observation = eval_env.reset()[0]
        done = False
        while not done:
            if episode == 0 and video_folder is not None:
                frames.append(eval_env.render())
            observation = np.reshape(observation, [1, observation_dimensions])
            action = agent.sample_action(observation)
            observation_new, reward, done, truncated, _ = eval_env.step(action)
            total_reward += reward
            if truncated:
                done = True
            observation_new = np.reshape(observation_new, [1, observation_dimensions])
            observation = observation_new
        if episode == 0 and video_folder is not None:
            save_video(frames, video_folder, fps=eval_env.metadata["render_fps"],
                       name_prefix=f"dqn_epoch_{current_epoch}")
    average_reward = total_reward / episodes

    return average_reward


def run_ppo_evaluation(agent, eval_env, episodes, video_folder, current_epoch):
    observation_dimensions = len(eval_env.reset()[0])
    total_reward = 0
    frames = []
    for episode in range(episodes):
        observation = eval_env.reset()[0]
        done = False
        while not done:
            if episode == 0 and video_folder is not None:
                frames.append(eval_env.render())
            observation = np.reshape(observation, [1, observation_dimensions])
            logits, action = agent.sample_action(observation)
            observation_new, reward, done, truncated, _ = eval_env.step(action[0].numpy())
            total_reward += reward
            if truncated:
                done = True
            observation = observation_new
        if episode == 0 and video_folder is not None:
            save_video(frames, video_folder, fps=eval_env.metadata["render_fps"],
                       name_prefix=f"ppo_epoch_{current_epoch}")
    average_reward = total_reward / episodes

    return average_reward


def run_rnd_evaluation(agent, eval_env, episodes, video_folder, current_epoch):
    observation_dimensions = len(eval_env.reset()[0])
    total_reward = 0
    total_intrinsic_reward = 0
    frames = []
    for episode in range(episodes):
        observation = eval_env.reset()[0]
        observation = np.reshape(observation, [1, observation_dimensions])
        done = False
        while not done:
            if episode == 0 and video_folder is not None:
                frames.append(eval_env.render())
            logits, action = agent.sample_action(observation)
            observation_new, extrinsic_reward, done, truncated, _ = eval_env.step(action[0].numpy())
            total_reward += extrinsic_reward
            if truncated:
                done = True
            observation_new = np.reshape(observation_new, [1, observation_dimensions])
            intrinsic_reward = agent.calculate_intrinsic_reward(observation_new)
            total_intrinsic_reward += intrinsic_reward
            observation = observation_new
        if episode == 0 and video_folder is not None:
            save_video(frames, video_folder, fps=eval_env.metadata["render_fps"],
                       name_prefix=f"rnd_epoch_{current_epoch}")
    average_reward = total_reward / episodes
    average_intrinsic_reward = total_intrinsic_reward / episodes

    return average_reward, average_intrinsic_reward


def load_file_for_plot(environment, algorithm, obj, date):
    with open(f'C:/Users/samjp/anaconda3/envs/msc-dissertation/msc_dissertation/dissertation_files/tests/test_data/{environment}/data/{algorithm}_{obj}_{date}.pkl', 'rb') as f:
        file = {algorithm: pickle.load(f)}
    return file


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


def get_all_visitable_cells(gridworld_env):
    grid = get_grid_representation(gridworld_env)
    visitable_cells = {}
    grid_height = len(grid)
    grid_width = len(grid[0])
    for j in range(grid_height):
        for i in range(grid_width):
            if grid[i, j] != 2:  # wall
                visitable_cells[(j, i)] = []
    return visitable_cells


def plot_exploration_heatmap(gridworld_env, first_time_visits, epochs, steps_per_epoch, env_name):
    algorithm, updated_ftvs = list(first_time_visits.items())[0]
    grid = get_grid_representation(gridworld_env)
    unvisitable_cells = get_unvisitable_cells(grid)
    updated_ftvs.update(unvisitable_cells)
    ser = pd.Series(list(updated_ftvs.values()),
                    index=pd.MultiIndex.from_tuples(updated_ftvs.keys()))
    df = ser.unstack().transpose().fillna(epochs*steps_per_epoch + 1)
    cmap = sns.color_palette("coolwarm", as_cmap=True)
    cmap.set_under("grey")  # walls
    cmap.set_over("black")  # unvisited states
    sns.heatmap(df, vmin=0, vmax=epochs*steps_per_epoch, cmap=cmap, xticklabels=False, yticklabels=False,
                cbar_kws={'label': 'Training Steps'})
    plt.title(f"Exploration heatmap - {algorithm}")
    plt.savefig(f'C:/Users/samjp/anaconda3/envs/msc-dissertation/msc_dissertation/dissertation_files/tests/test_data/{env_name}/plots/{algorithm}_heatmap')
    plt.show()


def plot_evaluation_data(rewards, epochs, eval_frequency, steps_per_epoch, env_name):
    step_list = [(i * steps_per_epoch * eval_frequency) for i in range((epochs // eval_frequency) + 1)]
    fig, ax = plt.subplots()
    for key in rewards.keys():
        arr = np.array(rewards[key])
        ave_reward = np.mean(arr, axis=0)
        if arr.ndim == 1:
            ax.plot(step_list, arr, label=key)
        else:
            ax.plot(step_list, ave_reward, label=key)
            conf_ints = [confidence_interval(i) for i in arr.T]
            ax.fill_between(step_list,
                            [ave_reward[i] - conf_ints[i] for i in range(len(ave_reward))],
                            [ave_reward[i] + conf_ints[i] for i in range(len(ave_reward))],
                            alpha=0.2)
    plt.xlim(0, steps_per_epoch * epochs)
    plt.ylim(0, 1)
    plt.ylabel("Average Extrinsic Reward")
    plt.xlabel("Training Steps")
    plt.title("Average reward after training for x steps")
    ax.legend(loc='upper left')
    plt.savefig(f'C:/Users/samjp/anaconda3/envs/msc-dissertation/msc_dissertation/dissertation_files/tests/test_data/{env_name}/plots/extrinsic_rewards')
    plt.show()


def plot_state_visit_percentage(gridworld_env, first_time_visits, epochs, steps_per_epoch, env_name):
    grid = get_grid_representation(gridworld_env)
    total_states = grid.shape[0] * grid.shape[1]
    unvisitable_cells = get_unvisitable_cells(grid)
    total_states -= len(unvisitable_cells)
    fig, ax = plt.subplots()
    for key in first_time_visits.keys():
        state_visit_steps = sorted(list(first_time_visits[key].values()))
        y = [((i + 1) / total_states) * 100 for i in range(len(state_visit_steps))]
        y.append(y[-1])
        state_visit_steps.append(steps_per_epoch * epochs)
        ax.plot(state_visit_steps, y, label=key)
    plt.ylabel("Percentage of grid cells explored")
    plt.xlabel("Training Steps")
    plt.xlim(0, steps_per_epoch * epochs)
    plt.ylim(0, 100)
    ax.legend(loc='lower right')
    plt.title("Percentage of the environment explored throughout training")
    plt.savefig(f'C:/Users/samjp/anaconda3/envs/msc-dissertation/msc_dissertation/dissertation_files/tests/test_data/{env_name}/plots/state_visit_percentage')
    plt.show()

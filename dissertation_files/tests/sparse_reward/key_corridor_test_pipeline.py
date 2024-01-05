import numpy as np
import warnings
import os
import pickle
import gymnasium as gym
import datetime as dt
from dissertation_files.agents import config
from dissertation_files.agents.agent import RandomAgent, DQNAgent, PPOAgent, RNDAgent
from dissertation_files.environments.minigrid_environments import RGBImgPartialObsWrapper
from dissertation_files.agents.training import random_play_loop, dqn_training_loop, ppo_training_loop, rnd_training_loop
from dissertation_files.agents.evaluation import get_all_visitable_cells


warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

"""
Training & Evaluation Hyperparameters
"""

STEPS_PER_EPOCH = 4000
EPOCHS = 200
EVALUATION_FREQUENCY = 10
EVALUATION_EPISODES_PER_EPOCH = 50
EVALUATION_PIPELINE_RUNS = 10

if __name__ == "__main__":

    """
    Environment Set Up
    """

    env = gym.make("MiniGrid-KeyCorridorS5R3", render_mode='rgb_array')
    env = RGBImgPartialObsWrapper(env, seed=1)
    eval_env = gym.make("MiniGrid-KeyCorridorS5R3", render_mode='rgb_array')
    eval_env = RGBImgPartialObsWrapper(eval_env, seed=1)
    observation_dimensions = len(env.reset()[0])
    action_dimensions = env.action_space.n

    print("=============================================")

    """
    Random Agent
    """

    rewards = []
    ftvs = get_all_visitable_cells(env)

    for i in range(EVALUATION_PIPELINE_RUNS):
        print(f"Random Agent Run {i+1}")

        if i == 0:
            video_folder = '../test_data/sparse_key_corridor/videos'
        else:
            video_folder = None

        agent = RandomAgent(
            action_dimensions
        )

        average_reward_list, first_time_visits = random_play_loop(
            EPOCHS,
            agent,
            env,
            STEPS_PER_EPOCH,
            video_folder=video_folder,
            eval_env=eval_env,
            eval_epoch_frequency=EVALUATION_FREQUENCY,
            eval_episodes_per_epoch=EVALUATION_EPISODES_PER_EPOCH
        )

        rewards.append(average_reward_list)
        for key in first_time_visits.keys():
            ftvs[key].append(first_time_visits[key])

        env.reset()
        eval_env.reset()
        print("=============================================")

    for key in ftvs.copy().keys():
        if ftvs[key] == []:
            ftvs.pop(key)
        elif len(ftvs[key]) < EVALUATION_PIPELINE_RUNS:
            while len(ftvs[key]) < EVALUATION_PIPELINE_RUNS:
                ftvs[key].append(EPOCHS * STEPS_PER_EPOCH)
            ftvs[key] = np.mean(ftvs[key])
        else:
            ftvs[key] = np.mean(ftvs[key])

    with open(rf'../test_data/sparse_key_corridor/data/random_rewards_{dt.date.today()}.pkl', 'wb+') as f:
        pickle.dump(rewards, f)

    with open(rf'../test_data/sparse_key_corridor/data/random_ftvs_{dt.date.today()}.pkl', 'wb+') as f:
        pickle.dump(ftvs, f)

    """
    DQN
    """

    rewards = []
    ftvs = get_all_visitable_cells(env)

    for i in range(EVALUATION_PIPELINE_RUNS):
        print(f"DQN Agent Run {i+1}")

        if i == 0:
            video_folder = '../test_data/sparse_key_corridor/videos'
        else:
            video_folder = None

        agent = DQNAgent(
            observation_dimensions,
            action_dimensions,
            config.mg_dqn_memory_size,
            config.mg_dqn_batch_size,
            config.mg_dqn_hidden_sizes,
            config.mg_dqn_input_activation,
            config.mg_dqn_output_activation,
            config.mg_dqn_learning_rate,
            config.mg_dqn_epsilon,
            config.mg_dqn_epsilon_decay,
            config.mg_dqn_min_epsilon,
            config.mg_dqn_gamma
        )

        average_reward_list, first_time_visits = dqn_training_loop(
            EPOCHS,
            agent,
            env,
            observation_dimensions,
            STEPS_PER_EPOCH,
            config.mg_dqn_steps_target_model_update,
            video_folder=video_folder,
            eval_env=eval_env,
            eval_epoch_frequency=EVALUATION_FREQUENCY,
            eval_episodes_per_epoch=EVALUATION_EPISODES_PER_EPOCH
        )

        rewards.append(average_reward_list)
        for key in first_time_visits.keys():
            ftvs[key].append(first_time_visits[key])

        env.reset()
        eval_env.reset()
        print("=============================================")

    for key in ftvs.copy().keys():
        if ftvs[key] == []:
            ftvs.pop(key)
        elif len(ftvs[key]) < EVALUATION_PIPELINE_RUNS:
            while len(ftvs[key]) < EVALUATION_PIPELINE_RUNS:
                ftvs[key].append(EPOCHS * STEPS_PER_EPOCH)
            ftvs[key] = np.mean(ftvs[key])
        else:
            ftvs[key] = np.mean(ftvs[key])

    with open(rf'../test_data/sparse_key_corridor/data/dqn_rewards_{dt.date.today()}.pkl', 'wb+') as f:
        pickle.dump(rewards, f)

    with open(rf'../test_data/sparse_key_corridor/data/dqn_ftvs_{dt.date.today()}.pkl', 'wb+') as f:
        pickle.dump(ftvs, f)


    """
    PPO
    """

    rewards = []
    ftvs = get_all_visitable_cells(env)

    for i in range(EVALUATION_PIPELINE_RUNS):
        print(f"PPO Agent Run {i+1}")

        if i == 0:
            video_folder = '../test_data/sparse_key_corridor/videos'
        else:
            video_folder = None

        agent = PPOAgent(
            observation_dimensions,
            action_dimensions,
            STEPS_PER_EPOCH,
            config.mg_ppo_hidden_sizes,
            config.mg_ppo_input_activation,
            config.mg_ppo_output_activation,
            config.mg_ppo_actor_learning_rate,
            config.mg_ppo_critic_learning_rate,
            config.mg_ppo_clip_ratio,
            config.mg_ppo_gamma,
            config.mg_ppo_lam
        )

        average_reward_list, first_time_visits = ppo_training_loop(
            EPOCHS,
            agent,
            env,
            observation_dimensions,
            action_dimensions,
            STEPS_PER_EPOCH,
            config.mg_ppo_train_actor_iterations,
            config.mg_ppo_train_critic_iterations,
            video_folder=video_folder,
            eval_env=eval_env,
            eval_epoch_frequency=EVALUATION_FREQUENCY,
            eval_episodes_per_epoch=EVALUATION_EPISODES_PER_EPOCH
        )

        rewards.append(average_reward_list)
        for key in first_time_visits.keys():
            ftvs[key].append(first_time_visits[key])

        env.reset()
        eval_env.reset()
        print("=============================================")

    for key in ftvs.copy().keys():
        if ftvs[key] == []:
            ftvs.pop(key)
        elif len(ftvs[key]) < EVALUATION_PIPELINE_RUNS:
            while len(ftvs[key]) < EVALUATION_PIPELINE_RUNS:
                ftvs[key].append(EPOCHS * STEPS_PER_EPOCH)
            ftvs[key] = np.mean(ftvs[key])
        else:
            ftvs[key] = np.mean(ftvs[key])

    with open(rf'../test_data/sparse_key_corridor/data/ppo_rewards_{dt.date.today()}.pkl', 'wb+') as f:
        pickle.dump(rewards, f)

    with open(rf'../test_data/sparse_key_corridor/data/ppo_ftvs_{dt.date.today()}.pkl', 'wb+') as f:
        pickle.dump(ftvs, f)

    """
    RND
    """

    rewards = []
    ftvs = get_all_visitable_cells(env)

    for i in range(EVALUATION_PIPELINE_RUNS):
        print(f"RND Agent Run {i+1}")

        if i == 0:
            video_folder = '../test_data/sparse_key_corridor/videos'
        else:
            video_folder = None

        agent = RNDAgent(observation_dimensions,
                         action_dimensions,
                         STEPS_PER_EPOCH,
                         config.mg_rnd_hidden_sizes,
                         config.mg_rnd_input_activation,
                         config.mg_rnd_output_activation,
                         config.mg_rnd_actor_learning_rate,
                         config.mg_rnd_critic_learning_rate,
                         config.mg_rnd_rnd_predictor_learning_rate,
                         config.mg_rnd_clip_ratio,
                         config.mg_rnd_gamma,
                         config.mg_rnd_lam,
                         config.mg_rnd_intrinsic_weight)

        average_reward_list, _, first_time_visits = rnd_training_loop(
            EPOCHS,
            agent,
            env,
            observation_dimensions,
            action_dimensions,
            STEPS_PER_EPOCH,
            config.mg_rnd_train_actor_iterations,
            config.mg_rnd_train_critic_iterations,
            config.mg_rnd_train_rnd_iterations,
            video_folder=video_folder,
            eval_env=eval_env,
            eval_epoch_frequency=EVALUATION_FREQUENCY,
            eval_episodes_per_epoch=EVALUATION_EPISODES_PER_EPOCH)

        rewards.append(average_reward_list)
        for key in first_time_visits.keys():
            ftvs[key].append(first_time_visits[key])

        env.reset()
        eval_env.reset()
        print("=============================================")

    for key in ftvs.copy().keys():
        if ftvs[key] == []:
            ftvs.pop(key)
        elif len(ftvs[key]) < EVALUATION_PIPELINE_RUNS:
            while len(ftvs[key]) < EVALUATION_PIPELINE_RUNS:
                ftvs[key].append(EPOCHS * STEPS_PER_EPOCH)
            ftvs[key] = np.mean(ftvs[key])
        else:
            ftvs[key] = np.mean(ftvs[key])

    with open(rf'../test_data/sparse_key_corridor/data/rnd_rewards_{dt.date.today()}.pkl', 'wb+') as f:
        pickle.dump(rewards, f)

    with open(rf'../test_data/sparse_key_corridor/data/rnd_ftvs_{dt.date.today()}.pkl', 'wb+') as f:
        pickle.dump(ftvs, f)

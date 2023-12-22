import numpy as np
import warnings
import os
import tensorflow as tf
import pickle
import gymnasium as gym
import datetime as dt
from dissertation_files.agents import config
from dissertation_files.agents.agent import RNDAgent, PretrainedRNDAgent
from dissertation_files.environments.minigrid_environments import RGBImgPartialObsWrapper
from dissertation_files.agents.training import rnd_training_loop
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
EVALUATION_PIPELINE_RUNS = 5

if __name__ == "__main__":

    """
    Environment Set Up
    """

    env = gym.make("MiniGrid-KeyCorridorS4R3", render_mode='rgb_array')
    env = RGBImgPartialObsWrapper(env, seed=1)
    eval_env = gym.make("MiniGrid-KeyCorridorS4R3", render_mode='rgb_array')
    eval_env = RGBImgPartialObsWrapper(eval_env, seed=1)
    observation_dimensions = len(env.reset()[0])
    action_dimensions = env.action_space.n

    print("=============================================")

    """
    RND
    """

    rewards = []
    ftvs = get_all_visitable_cells(env)

    for i in range(EVALUATION_PIPELINE_RUNS):
        print(f"RND Agent Run {i+1}")

        if i == 0:
            video_folder = '../test_data/generalisation_key_corridor_S4R3/videos'
        else:
            video_folder = None

        agent = PretrainedRNDAgent(
            observation_dimensions,
            action_dimensions,
            STEPS_PER_EPOCH,
            config.mg_rnd_hidden_sizes,
            config.mg_rnd_input_activation,
            config.mg_rnd_output_activation,
            config.mg_rnd_actor_learning_rate,
            config.mg_rnd_critic_learning_rate,
            config.mg_rnd_rnd_predictor_learning_rate * 0.5,
            config.mg_rnd_clip_ratio,
            config.mg_rnd_gamma,
            config.mg_rnd_lam,
            config.mg_rnd_intrinsic_weight * 0.1,
            'C:/Users/samjp/anaconda3/envs/msc-dissertation/msc_dissertation/dissertation_files/tests/test_data/checkpoints/key_corridor_S3R3',
            restore_rnd_networks=False
        )


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

    with open(rf'../test_data/generalisation_key_corridor_S4R3/data/rnd_rewards_{dt.date.today()}.pkl', 'wb+') as f:
        pickle.dump(rewards, f)

    with open(rf'../test_data/generalisation_key_corridor_S4R3/data/rnd_ftvs_{dt.date.today()}.pkl', 'wb+') as f:
        pickle.dump(ftvs, f)

import numpy as np
import warnings
import os
import pickle
import gymnasium as gym
import tensorflow as tf
import datetime as dt
from dissertation_files.agents import config
from dissertation_files.agents.agent import RandomAgent, DQNAgent, PPOAgent, RNDAgent
from dissertation_files.environments.minigrid_environments import RGBImgPartialObsWrapper, MultiroomFourRooms
from dissertation_files.agents.training import random_play_loop, dqn_training_loop, ppo_training_loop, rnd_training_loop
from dissertation_files.agents.evaluation import get_all_visitable_cells


warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

"""
Training & Evaluation Hyperparameters
"""

STEPS_PER_EPOCH = 4000
EPOCHS = 100
ENVIRONMENT_INITIALISATIONS = 1

if __name__ == "__main__":

    """
    Environment Set Up
    """

    env = MultiroomFourRooms(render_mode='rgb_array')
    env = RGBImgPartialObsWrapper(env)
    observation_dimensions = len(env.reset()[0])
    action_dimensions = env.action_space.n

    agent = RNDAgent(observation_dimensions,
                     action_dimensions,
                     STEPS_PER_EPOCH,
                     config.mg_rnd_hidden_sizes_generalisation,
                     config.mg_rnd_input_activation,
                     config.mg_rnd_output_activation,
                     config.mg_rnd_actor_learning_rate,
                     config.mg_rnd_critic_learning_rate,
                     config.mg_rnd_rnd_predictor_learning_rate,
                     config.mg_rnd_clip_ratio,
                     config.mg_rnd_gamma,
                     config.mg_rnd_lam,
                     config.mg_rnd_intrinsic_weight * 0.1)

    print("=============================================")

    for i in range(ENVIRONMENT_INITIALISATIONS):

        print(f"Environment Run {i+1}")

        #env.seed = i+1
        env.reset()

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
            video_folder=None,
            eval_env=None
        )

        average_reward_list.insert(0, 0)

        with open(rf'../test_data/generalisation_multiroom_N4/data/rnd_rewards_{dt.date.today()}.pkl', 'wb+') as f:
            pickle.dump(average_reward_list, f)

        with open(rf'../test_data/generalisation_multiroom_N4/data/rnd_ftvs_{dt.date.today()}.pkl', 'wb+') as f:
            pickle.dump(first_time_visits, f)

    print("=============================================")

    agent.save_models('C:/Users/samjp/anaconda3/envs/msc-dissertation/msc_dissertation/dissertation_files/tests/test_data/checkpoints/multiroom_N4')

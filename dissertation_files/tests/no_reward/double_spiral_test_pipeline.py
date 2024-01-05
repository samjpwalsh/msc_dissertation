import warnings
import os
import pickle
import datetime as dt
from dissertation_files.agents import config
from dissertation_files.agents.agent import RandomAgent, DQNAgent, PPOAgent, RNDAgent
from dissertation_files.environments.minigrid_environments import RGBImgPartialObsWrapper, DoubleSpiralMaze
from dissertation_files.agents.training import random_play_loop, dqn_training_loop, ppo_training_loop, rnd_training_loop


warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

"""
Training & Evaluation Hyperparameters
"""

STEPS_PER_EPOCH = 4000
EPOCHS = 200
EVALUATION_PIPELINE_RUNS = 10

if __name__ == "__main__":

    """
    Environment Set Up
    """

    env = DoubleSpiralMaze(render_mode='rgb_array')
    env = RGBImgPartialObsWrapper(env)
    observation_dimensions = len(env.reset()[0])
    action_dimensions = env.action_space.n

    print("=============================================")

    """
    Random Agent
    """

    for i in range(EVALUATION_PIPELINE_RUNS):
        print(f"Random Agent Run {i+1}")

        agent = RandomAgent(
            action_dimensions
        )

        _, first_time_visits = random_play_loop(
            EPOCHS,
            agent,
            env,
            STEPS_PER_EPOCH,
            video_folder=None,
            eval_env=None
        )

        with open(rf'../test_data/no_double_spiral/data/random_ftvs_{dt.date.today()}_run_{i+1}.pkl', 'wb+') as f:
            pickle.dump(first_time_visits, f)

        env.reset()
        print("=============================================")

    """
    DQN
    """

    for i in range(EVALUATION_PIPELINE_RUNS):
        print(f"DQN Agent Run {i+1}")

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

        _, first_time_visits = dqn_training_loop(
            EPOCHS,
            agent,
            env,
            observation_dimensions,
            STEPS_PER_EPOCH,
            config.mg_dqn_steps_target_model_update,
            video_folder=None,
            eval_env=None
        )

        with open(rf'../test_data/no_double_spiral/data/dqn_ftvs_{dt.date.today()}_run_{i+1}.pkl', 'wb+') as f:
            pickle.dump(first_time_visits, f)

        env.reset()
        print("=============================================")


    """
    PPO
    """

    for i in range(EVALUATION_PIPELINE_RUNS):
        print(f"PPO Agent Run {i+1}")

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

        _, first_time_visits = ppo_training_loop(
            EPOCHS,
            agent,
            env,
            observation_dimensions,
            action_dimensions,
            STEPS_PER_EPOCH,
            config.mg_ppo_train_actor_iterations,
            config.mg_ppo_train_critic_iterations,
            video_folder=None,
            eval_env=None
        )

        with open(rf'../test_data/no_double_spiral/data/ppo_ftvs_{dt.date.today()}_run_{i+1}.pkl', 'wb+') as f:
            pickle.dump(first_time_visits, f)

        env.reset()
        print("=============================================")

    """
    RND
    """

    for i in range(EVALUATION_PIPELINE_RUNS):
        print(f"RND Agent Run {i+1}")

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

        _, _, first_time_visits = rnd_training_loop(
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

        with open(rf'../test_data/no_double_spiral/data/rnd_ftvs_{dt.date.today()}_run_{i+1}.pkl', 'wb+') as f:
            pickle.dump(first_time_visits, f)

        env.reset()
        print("=============================================")

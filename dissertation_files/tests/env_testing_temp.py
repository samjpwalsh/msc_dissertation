import numpy as np
import warnings
import os
import gymnasium as gym
import pickle
import datetime as dt
from dissertation_files.agents import config
from dissertation_files.agents.agent import RandomAgent, DQNAgent, PPOAgent, RNDAgent
from dissertation_files.environments.minigrid_environments import TwoRooms
from dissertation_files.environments.minigrid_wrappers import FlatObsWrapper
from dissertation_files.agents.training import random_play_loop, dqn_training_loop, ppo_training_loop, rnd_training_loop
from dissertation_files.agents.evaluation import get_all_visitable_cells
from minigrid.manual_control import ManualControl


warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

"""
Training & Evaluation Hyperparameters
"""

STEPS_PER_EPOCH = 4000
EPOCHS = 30

"""
Environment Set Up
"""

env = TwoRooms(render_mode='human')
env = FlatObsWrapper(env)
observation_dimensions = len(env.reset()[0])
action_dimensions = env.action_space.n

manual_control = ManualControl(env, seed=42)
manual_control.start()
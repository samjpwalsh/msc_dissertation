import warnings
import os
from dissertation_files.environments.minigrid_environments import SpiralMaze, FlatObsWrapper, RGBImgPartialObsWrapper, SparseLockedRooms, SparseSequentialRooms, MultiroomFourRooms
from minigrid.manual_control import ManualControl
import matplotlib.pyplot as plt
import gymnasium as gym

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

env = MultiroomFourRooms(render_mode='human')
env = RGBImgPartialObsWrapper(env)

env.reset()

observation_dimensions = len(env.reset()[0])
action_dimensions = env.action_space.n

manual_control = ManualControl(env)
manual_control.start()
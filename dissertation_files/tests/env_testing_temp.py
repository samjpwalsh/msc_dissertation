import warnings
import os
from dissertation_files.environments.minigrid_environments import SpiralMaze, FlatObsWrapper, RGBImgPartialObsWrapper
from minigrid.manual_control import ManualControl
import matplotlib.pyplot as plt

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

env = SpiralMaze(render_mode='human')
env = RGBImgPartialObsWrapper(env)
obs, _ = env.reset()
obs = obs["image"]
plt.imshow(obs["image"])
plt.show()
observation_dimensions = len(env.reset()[0])
action_dimensions = env.action_space.n

manual_control = ManualControl(env, seed=42)
manual_control.start()
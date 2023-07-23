import numpy as np
import gymnasium as gym
from gymnasium import spaces
from minigrid.wrappers import ObservationWrapper
from dissertation_files.environments.simple_env import SimpleEnv
from dissertation_files.agents.agent import RandomAgent


class FlatObsWrapper(ObservationWrapper):
    # Transforms the observation into only the image component scaled to the grid size. Still partially observable,
    # direction component not included.

    def __init__(self, env):
        super().__init__(env)

        self.flat_grid_size = self.env.width * self.env.height * 3

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.flat_grid_size,),
            dtype="uint8"
        )

    def observation(self, obs):
        grid = obs["image"]

        # Flatten the visible grid and concatenate with one-hot encoded direction
        obs = grid.flatten()

        return obs


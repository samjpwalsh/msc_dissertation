import numpy as np
import gymnasium as gym
from gymnasium import spaces
from minigrid.wrappers import ObservationWrapper
from dissertation_files.environments.minigrid_environments import SimpleEnv
from dissertation_files.agents.agent import RandomAgent


class FlatObsWrapper(ObservationWrapper):
    # Transforms the observation into only the image component scaled to the grid size, and removes the colour component.
    # Still partially observable, direction component not included.

    def __init__(self, env):
        super().__init__(env)

        self.flat_grid_size = self.env.width * self.env.height * 2

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.flat_grid_size,),
            dtype="uint8"
        )

    def observation(self, obs):
        grid = obs["image"]
        new_grid = []
        for i in grid:
            new_row = []
            for j in i:
                new_j = [j[0], j[2]]
                new_row.append(new_j)
            new_grid.append(new_row)
        new_grid = np.array(new_grid)
        obs = new_grid.flatten()

        return obs


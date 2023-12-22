from __future__ import annotations
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.minigrid_env import MiniGridEnv
from enum import IntEnum
from gymnasium import spaces
from minigrid.wrappers import ObservationWrapper
from gymnasium.core import ActType, ObsType
from typing import Any, Iterable, SupportsFloat, TypeVar
import numpy as np
from dissertation_files.agents import config
from keras.models import Sequential
from keras.layers import Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation, Input
import tensorflow as tf

class DoorKeyActions(IntEnum):
    # Turn left, turn right, move forward
    left = 0
    right = 1
    forward = 2
    # Pick up an object
    pickup = 3
    # Toggle/activate an object
    toggle = 4

class TwoRoomsActions(IntEnum):
    # Turn left, turn right, move forward
    left = 0
    right = 1
    forward = 2
    # Toggle/activate an object
    toggle = 3


class SpiralMazeActions(IntEnum):
    # Turn left, turn right, move forward
    left = 0
    right = 1
    forward = 2

class SimpleEnvActions(IntEnum):
    # Turn left, turn right, move forward
    left = 0
    right = 1
    forward = 2


class SimpleEnv(MiniGridEnv):
    def __init__(
            self,
            size=15,
            agent_start_pos=(1, 1),
            agent_start_dir=0,
            max_steps=1000,
            **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(mission_func=self._gen_mission)

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=False,
            max_steps=max_steps,
            **kwargs,
        )

        self.actions = SimpleEnvActions
        self.action_space = spaces.Discrete(len(self.actions))

    @staticmethod
    def _gen_mission():
        return "simple env"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Place the agent
        self.agent_pos = self.agent_start_pos
        self.agent_dir = self.agent_start_dir

        self.mission = "simple env"

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.step_count += 1

        reward = 0
        terminated = False
        truncated = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = tuple(fwd_pos)
            if fwd_cell is not None and fwd_cell.type == "goal":
                terminated = True
                reward = self._reward()
            if fwd_cell is not None and fwd_cell.type == "lava":
                terminated = True

        else:
            raise ValueError(f"Unknown action: {action}")

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        obs = self.gen_obs()

        reward *= 100  # modify reward to give stronger signal

        return obs, reward, terminated, truncated, {}


class DoorKeyEnv(MiniGridEnv):
    def __init__(
            self,
            size=15,
            agent_start_pos=(1, 1),
            agent_start_dir=0,
            max_steps=1000,
            **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(mission_func=self._gen_mission)

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=False,
            max_steps=max_steps,
            **kwargs,
        )

        self.actions = DoorKeyActions
        self.action_space = spaces.Discrete(len(self.actions))

    @staticmethod
    def _gen_mission():
        return "door key"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # add wall, door and key
        for i in range(0, height):
            self.grid.set(8, i, Wall())

        self.grid.set(8, 6, Door(COLOR_NAMES[0], is_locked=True))
        self.grid.set(4, 10, Key(COLOR_NAMES[0]))

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Place the agent
        self.agent_pos = self.agent_start_pos
        self.agent_dir = self.agent_start_dir

        self.mission = "door key"

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.step_count += 1

        reward = 0
        terminated = False
        truncated = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = tuple(fwd_pos)
            if fwd_cell is not None and fwd_cell.type == "goal":
                terminated = True
                reward = self._reward()
            if fwd_cell is not None and fwd_cell.type == "lava":
                terminated = True

        # Pick up an object
        elif action == self.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if self.carrying is None:
                    self.carrying = fwd_cell
                    self.carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(fwd_pos[0], fwd_pos[1], None)

        # Toggle/activate an object
        elif action == self.actions.toggle:
            if fwd_cell:
                fwd_cell.toggle(self, fwd_pos)

        else:
            raise ValueError(f"Unknown action: {action}")

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        obs = self.gen_obs()

        reward *= 100  # modify reward to give stronger signal

        return obs, reward, terminated, truncated, {}

class TwoRooms(MiniGridEnv):
    def __init__(
            self,
            size=15,
            agent_start_pos=(7, 13),
            agent_start_dir=3,
            max_steps=1000,
            **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(mission_func=self._gen_mission)

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=False,
            max_steps=max_steps,
            **kwargs,
        )

        self.actions = TwoRoomsActions
        self.action_space = spaces.Discrete(len(self.actions))

    @staticmethod
    def _gen_mission():
        return "two rooms"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # add room 1
        for i in range(0, 6):
            self.grid.set(5, i, Wall())
        for i in range(0, 6):
            self.grid.set(i, 6, Wall())
        self.grid.set(2, 6, Door(COLOR_NAMES[0], is_locked=False))

        # add room 2
        for i in range(0, 6):
            self.grid.set(9, i, Wall())
        for i in range(9, 14):
            self.grid.set(i, 6, Wall())
        self.grid.set(12, 6, Door(COLOR_NAMES[0], is_locked=False))


        # Place a goal square in room 2
        self.put_obj(Goal(), 11, 2)

        # Place the agent
        self.agent_pos = self.agent_start_pos
        self.agent_dir = self.agent_start_dir

        self.mission = "two rooms"

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.step_count += 1

        reward = 0
        terminated = False
        truncated = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = tuple(fwd_pos)
            if fwd_cell is not None and fwd_cell.type == "goal":
                terminated = True
                reward = self._reward()
            if fwd_cell is not None and fwd_cell.type == "lava":
                terminated = True

        # Toggle/activate an object
        elif action == self.actions.toggle:
            if fwd_cell:
                fwd_cell.toggle(self, fwd_pos)

        else:
            raise ValueError(f"Unknown action: {action}")

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        obs = self.gen_obs()

        reward *= 100  # modify reward to give stronger signal

        return obs, reward, terminated, truncated, {}


class SpiralMaze(MiniGridEnv):
    def __init__(
            self,
            size=15,
            agent_start_pos=(6, 1),
            agent_start_dir=1,
            max_steps=200,
            **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(mission_func=self._gen_mission)

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=False,
            max_steps=max_steps,
            **kwargs,
        )

        self.actions = SpiralMazeActions
        self.action_space = spaces.Discrete(len(self.actions))

    @staticmethod
    def _gen_mission():
        return "spiral maze"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        for i in range(1, 14):
            self.grid.set(1, i, Wall())
            self.grid.set(13, i, Wall())

        # build the spiral
        for i in range(3, 14):
            self.grid.set(2, i, Wall())
            self.grid.set(12, i, Wall())
        for j in range(1, 4):
            for i in range(2, 6):
                self.grid.set(i, j, Wall())
            for i in range(9, 13):
                self.grid.set(i, j, Wall())
        for i in range(2, 11):
            self.grid.set(i, 5, Wall())
        for i in range(6, 13):
            self.grid.set(10, i, Wall())
        for i in range(4, 10):
            self.grid.set(i, 12, Wall())
        for i in range(7, 12):
            self.grid.set(4, i, Wall())
        for i in range(5, 9):
            self.grid.set(i, 7, Wall())
        for i in range(8, 11):
            self.grid.set(8, i, Wall())
        for i in range(6, 8):
            self.grid.set(i, 10, Wall())
        self.grid.set(6, 9, Wall())

        # Place a goal square in room 2
        self.put_obj(Goal(), 7, 9)

        # Place the agent
        self.agent_pos = self.agent_start_pos
        self.agent_dir = self.agent_start_dir

        self.mission = "spiral maze"

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.step_count += 1

        reward = 0
        terminated = False
        truncated = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = tuple(fwd_pos)
            if fwd_cell is not None and fwd_cell.type == "goal":
                terminated = True
                reward = self._reward()
            if fwd_cell is not None and fwd_cell.type == "lava":
                terminated = True

        else:
            raise ValueError(f"Unknown action: {action}")

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        obs = self.gen_obs()

        return obs, reward, terminated, truncated, {}

class DoubleSpiralMaze(MiniGridEnv):
    def __init__(
            self,
            size=21,
            agent_start_pos=(10, 5),
            agent_start_dir=1,
            max_steps=200,
            **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(mission_func=self._gen_mission)

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=False,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "double spiral maze"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        for i in range(1, 20):
            for j in range(1, 5):
                self.grid.set(i, j, Wall())
            for j in range(15, 20):
                self.grid.set(i, j, Wall())

        # build the spirals
        for i in range(2, 19):
            self.grid.set(i, 6, Wall())
        for j in range(6, 14):
            self.grid.set(2, j, Wall())
            self.grid.set(18, j, Wall())
        for i in range(2, 9):
            self.grid.set(i, 13, Wall())
        for i in range(12, 19):
            self.grid.set(i, 13, Wall())
        for j in range(8, 13):
            self.grid.set(8, j, Wall())
            self.grid.set(12, j, Wall())
        for i in range(4, 8):
            self.grid.set(i, 8, Wall())
        for i in range(13, 17):
            self.grid.set(i, 8, Wall())
        for j in range(9, 12):
            self.grid.set(4, j, Wall())
            self.grid.set(16, j, Wall())
        for i in range(5, 7):
            self.grid.set(i, 11, Wall())
        for i in range(14, 16):
            self.grid.set(i, 11, Wall())
        self.grid.set(6, 10, Wall())
        self.grid.set(14, 10, Wall())
        for j in range(7, 15):
            self.grid.set(10, j, Wall())

        # Place the agent
        self.agent_pos = self.agent_start_pos
        self.agent_dir = self.agent_start_dir

        self.mission = "double spiral maze"

class QuadSpiralMaze(MiniGridEnv):
    def __init__(
            self,
            size=21,
            agent_start_pos=(10, 10),
            agent_start_dir=1,
            max_steps=200,
            **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(mission_func=self._gen_mission)

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=False,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "quad spiral maze"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # build the spirals
        for i in range(2, 19):
            self.grid.set(i, 11, Wall())
            self.grid.set(i, 9, Wall())
        for j in range(11, 19):
            self.grid.set(2, j, Wall())
            self.grid.set(18, j, Wall())
        for j in range(2, 10):
            self.grid.set(2, j, Wall())
            self.grid.set(18, j, Wall())
        for i in range(2, 9):
            self.grid.set(i, 18, Wall())
            self.grid.set(i, 2, Wall())
        for i in range(12, 19):
            self.grid.set(i, 18, Wall())
            self.grid.set(i, 2, Wall())
        for j in range(13, 18):
            self.grid.set(8, j, Wall())
            self.grid.set(12, j, Wall())
        for j in range(2, 8):
            self.grid.set(8, j, Wall())
            self.grid.set(12, j, Wall())
        for i in range(4, 8):
            self.grid.set(i, 13, Wall())
        for i in range(13, 17):
            self.grid.set(i, 13, Wall())
        for i in range(4, 8):
            self.grid.set(i, 7, Wall())
        for i in range(13, 17):
            self.grid.set(i, 7, Wall())
        for j in range(14, 17):
            self.grid.set(4, j, Wall())
            self.grid.set(16, j, Wall())
        for j in range(4, 7):
            self.grid.set(4, j, Wall())
            self.grid.set(16, j, Wall())
        for i in range(5, 7):
            self.grid.set(i, 16, Wall())
        for i in range(14, 16):
            self.grid.set(i, 16, Wall())
        for i in range(5, 7):
            self.grid.set(i, 4, Wall())
        for i in range(14, 16):
            self.grid.set(i, 4, Wall())
        self.grid.set(6, 15, Wall())
        self.grid.set(14, 15, Wall())
        self.grid.set(6, 5, Wall())
        self.grid.set(14, 5, Wall())
        for j in range(1, 9):
            self.grid.set(10, j, Wall())
        for j in range(12, 20):
            self.grid.set(10, j, Wall())

        # Place the agent
        self.agent_pos = self.agent_start_pos
        self.agent_dir = self.agent_start_dir

        self.mission = "quad spiral maze"

class SparseLockedRooms(MiniGridEnv):
    def __init__(
            self,
            size=10,
            agent_start_pos=(7, 7),
            agent_start_dir=3,
            max_steps=300,
            **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(mission_func=self._gen_mission)

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=False,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "Sparse Reward - Locked Rooms"

    def _gen_grid(self, width, height):

        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        for i in range(1, 5):
            self.grid.set(4, i, Wall())
            self.grid.set(i, 4, Wall())
            self.grid.set(5, i, Wall())
        for i in range(6, 9):
            self.grid.set(i, 4, Wall())

        self.grid.set(2, 4, Door(COLOR_NAMES[0], is_locked=True))
        self.grid.set(7, 4, Door(COLOR_NAMES[1], is_locked=True))
        self.grid.set(2, 7, Key(color=COLOR_NAMES[0]))
        self.grid.set(2, 8, Key(color=COLOR_NAMES[1]))

        self.put_obj(Goal(), 7, 2)

        # Place the agent
        self.agent_pos = self.agent_start_pos
        self.agent_dir = self.agent_start_dir

        self.mission = "Sparse Reward - Locked Rooms"


class SparseSequentialRooms(MiniGridEnv):
    def __init__(
            self,
            size=13,
            agent_start_pos=(2, 2),
            agent_start_dir=3,
            max_steps=200,
            **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(mission_func=self._gen_mission)

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=False,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "Sparse Reward - Sequential Rooms"

    def _gen_grid(self, width, height):

        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        for i in range(1, 4):
            self.grid.set(4, i, Wall())
            self.grid.set(8, i, Wall())
        for i in range(0, 13):
            self.grid.set(i, 4, Wall())

        self.grid.set(4, 2, Door(COLOR_NAMES[0], is_locked=False))
        self.grid.set(8, 3, Door(COLOR_NAMES[1], is_locked=False))

        self.put_obj(Goal(), 11, 1)

        # Place the agent
        self.agent_pos = self.agent_start_pos
        self.agent_dir = self.agent_start_dir

        self.mission = "Sparse Reward - Sequential Rooms"

class MultiroomFourRooms(MiniGridEnv):

    def __init__(
            self,
            size=25,
            agent_start_pos=(8, 1),
            agent_start_dir=1,
            max_steps=100,
            **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(mission_func=self._gen_mission)

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=False,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "Generalisation - Multiroom"

    def _gen_grid(self, width, height):

        self.grid = Grid(width, height)

        # Room 1
        for i in range(3, 10):
            self.grid.set(i, 0, Wall())
            self.grid.set(i, 5, Wall())
        for j in range(1, 6):
            self.grid.set(3, j, Wall())
            self.grid.set(9, j, Wall())

        # Room 2
        for i in range(9, 14):
            self.grid.set(i, 2, Wall())
            self.grid.set(i, 6, Wall())
        for j in range(2, 7):
            self.grid.set(9, j, Wall())
            self.grid.set(13, j, Wall())

        # Room 3
        for i in range(14, 22):
            self.grid.set(i, 3, Wall())
            self.grid.set(i, 7, Wall())
        for j in range(3, 8):
            self.grid.set(13, j, Wall())
            self.grid.set(21, j, Wall())

        # Room 4
        for i in range(15, 19):
            self.grid.set(i, 14, Wall())
        for j in range(8, 15):
            self.grid.set(15, j, Wall())
            self.grid.set(18, j, Wall())

        self.grid.set(9, 3, Door(COLOR_NAMES[3], is_locked=False))
        self.grid.set(13, 5, Door(COLOR_NAMES[4], is_locked=False))
        self.grid.set(17, 7, Door(COLOR_NAMES[0], is_locked=False))

        self.put_obj(Goal(), 16, 13)

        # Place the agent
        self.agent_pos = self.agent_start_pos
        self.agent_dir = self.agent_start_dir

        self.mission = "Generalisation - Multiroom"

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


class RGBImgPartialObsWrapper(ObservationWrapper):
    """
    Wrapper to use partially observable RGB image as observation.
    This can be used to have the agent to solve the gridworld in pixel space.
    """

    def __init__(self, env, seed=None, tile_size=8):
        super().__init__(env)
        self.seed = seed
        tf.random.set_seed(1)

        # Rendering attributes for observations
        self.tile_size = tile_size

        obs_shape = env.observation_space.spaces["image"].shape
        self.new_image_space = spaces.Box(
            low=0,
            high=255,
            shape=(obs_shape[0] * tile_size, obs_shape[1] * tile_size, 3),
            dtype="uint8",
        )

        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "image": self.new_image_space}
        )

        self.conv_model = Sequential([
            Input(shape=self.new_image_space.shape),
            Conv2D(config.mg_conv_layers[0], kernel_size=config.mg_kernel_size[0], strides=config.mg_strides[0], activation=config.mg_conv_hidden_activation),
            Conv2D(config.mg_conv_layers[1], kernel_size=config.mg_kernel_size[1], strides=config.mg_strides[1], activation=config.mg_conv_hidden_activation),
            Flatten()
            ])

    def observation(self, obs):
        rgb_img_partial = self.get_frame(tile_size=self.tile_size, agent_pov=True) / 255
        rgb_img_partial = rgb_img_partial[None, ...]
        new_obs = self.conv_model(rgb_img_partial).numpy()[0]

        return new_obs

    def reset(self, seed=None):
        return super().reset(seed=self.seed)
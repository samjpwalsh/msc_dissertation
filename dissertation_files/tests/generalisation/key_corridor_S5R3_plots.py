from dissertation_files.agents.evaluation import plot_evaluation_data, plot_exploration_heatmap, load_file_for_plot
from dissertation_files.environments.minigrid_environments import RGBImgPartialObsWrapper
import key_corridor_S5R3_test_pipeline
import gymnasium as gym

env = gym.make("MiniGrid-KeyCorridorS5R3", render_mode='rgb_array')
env = RGBImgPartialObsWrapper(env, seed=1)
env.reset()
EPOCHS = key_corridor_S5R3_test_pipeline.EPOCHS
STEPS_PER_EPOCH = key_corridor_S5R3_test_pipeline.STEPS_PER_EPOCH
EVALUATION_FREQUENCY = key_corridor_S5R3_test_pipeline.EVALUATION_FREQUENCY

"""
Load data
"""

rnd_rewards = load_file_for_plot('generalisation_key_corridor_S5R3', 'rnd', 'rewards', '2023-12-29')
rnd_ftvs = load_file_for_plot('generalisation_key_corridor_S5R3', 'rnd', 'ftvs', '2023-12-29')

"""
Reward
"""

plot_evaluation_data(rnd_rewards, EPOCHS, EVALUATION_FREQUENCY, STEPS_PER_EPOCH, 'generalisation_key_corridor_S5R3')

"""
Heatmaps
"""

plot_exploration_heatmap(env, rnd_ftvs, EPOCHS, STEPS_PER_EPOCH, 'generalisation_key_corridor_S5R3')

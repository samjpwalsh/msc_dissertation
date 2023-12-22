from dissertation_files.agents.evaluation import plot_evaluation_data, plot_exploration_heatmap, \
    plot_state_visit_percentage, load_file_for_plot
from dissertation_files.environments.minigrid_environments import SparseSequentialRooms, FlatObsWrapper, RGBImgPartialObsWrapper
import multiroom_test_pipeline
import gymnasium as gym

env = gym.make("MiniGrid-MultiRoom-N6-v0", render_mode='rgb_array')
env = RGBImgPartialObsWrapper(env, seed=1)
env.reset()
EPOCHS = multiroom_test_pipeline.EPOCHS
STEPS_PER_EPOCH = multiroom_test_pipeline.STEPS_PER_EPOCH
EVALUATION_FREQUENCY = multiroom_test_pipeline.EVALUATION_FREQUENCY

"""
Load data
"""

rnd_rewards = load_file_for_plot('generalisation_multiroom_N6', 'rnd', 'rewards', '2023-12-22')
rnd_ftvs = load_file_for_plot('generalisation_multiroom_N6', 'rnd', 'ftvs', '2023-12-22')

"""
Reward
"""

plot_evaluation_data(rnd_rewards, EPOCHS, EVALUATION_FREQUENCY, STEPS_PER_EPOCH, 'generalisation_multiroom_N6')

"""
Heatmaps
"""

plot_exploration_heatmap(env, rnd_ftvs, EPOCHS, STEPS_PER_EPOCH, 'generalisation_multiroom_N6')

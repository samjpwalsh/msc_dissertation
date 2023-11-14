from dissertation_files.agents.evaluation import plot_evaluation_data, plot_exploration_heatmap, \
    plot_state_visit_percentage, load_file_for_plot
from dissertation_files.environments.minigrid_environments import SimpleEnv, FlatObsWrapper, RGBImgPartialObsWrapper
import key_corridor_test_pipeline
import gymnasium as gym

env = gym.make("MiniGrid-KeyCorridorS3R3", render_mode='rgb_array')
env = RGBImgPartialObsWrapper(env, seed=1)
env.reset()
EPOCHS = key_corridor_test_pipeline.EPOCHS
STEPS_PER_EPOCH = key_corridor_test_pipeline.STEPS_PER_EPOCH
EVALUATION_FREQUENCY = key_corridor_test_pipeline.EVALUATION_FREQUENCY

"""
Load data
"""

rnd_rewards = load_file_for_plot('key_corridor', 'rnd', 'rewards', '2023-11-10')
rnd_ftvs = load_file_for_plot('key_corridor', 'rnd', 'ftvs', '2023-11-10')

"""
Reward
"""

plot_evaluation_data(rnd_rewards, EPOCHS, EVALUATION_FREQUENCY, STEPS_PER_EPOCH, 'simple_env')

"""
State visit %
"""

plot_state_visit_percentage(env, rnd_ftvs, EPOCHS, STEPS_PER_EPOCH, 'simple_env')

"""
Heatmaps
"""

plot_exploration_heatmap(env, rnd_ftvs, EPOCHS, STEPS_PER_EPOCH, 'simple_env')

from dissertation_files.agents.evaluation import plot_evaluation_data, plot_exploration_heatmap, load_file_for_plot
from dissertation_files.environments.minigrid_environments import RGBImgPartialObsWrapper
import key_corridor_test_pipeline
import gymnasium as gym

env = gym.make("MiniGrid-KeyCorridorS5R3", render_mode='rgb_array')
env = RGBImgPartialObsWrapper(env, seed=1)
env.reset()
EPOCHS = key_corridor_test_pipeline.EPOCHS
STEPS_PER_EPOCH = key_corridor_test_pipeline.STEPS_PER_EPOCH
EVALUATION_FREQUENCY = key_corridor_test_pipeline.EVALUATION_FREQUENCY

"""
Load data
"""

random_rewards = load_file_for_plot('sparse_key_corridor', 'random', 'rewards', '2023-11-30')
dqn_rewards = load_file_for_plot('sparse_key_corridor', 'dqn', 'rewards', '2023-11-29')
ppo_rewards = load_file_for_plot('sparse_key_corridor', 'ppo', 'rewards', '2023-11-29')
rnd_rewards = load_file_for_plot('sparse_key_corridor', 'rnd', 'rewards', '2023-11-28')

random_ftvs = load_file_for_plot('sparse_key_corridor', 'random', 'ftvs', '2023-11-30')
dqn_ftvs = load_file_for_plot('sparse_key_corridor', 'dqn', 'ftvs', '2023-11-29')
ppo_ftvs = load_file_for_plot('sparse_key_corridor', 'ppo', 'ftvs', '2023-11-29')
rnd_ftvs = load_file_for_plot('sparse_key_corridor', 'rnd', 'ftvs', '2023-11-28')

"""
Reward
"""

rewards = random_rewards | dqn_rewards | ppo_rewards | rnd_rewards
plot_evaluation_data(rewards, EPOCHS, EVALUATION_FREQUENCY, STEPS_PER_EPOCH, 'sparse_key_corridor')

"""
Heatmaps
"""

plot_exploration_heatmap(env, random_ftvs, EPOCHS, STEPS_PER_EPOCH, 'sparse_key_corridor')
plot_exploration_heatmap(env, dqn_ftvs, EPOCHS, STEPS_PER_EPOCH, 'sparse_key_corridor')
plot_exploration_heatmap(env, ppo_ftvs, EPOCHS, STEPS_PER_EPOCH, 'sparse_key_corridor')
plot_exploration_heatmap(env, rnd_ftvs, EPOCHS, STEPS_PER_EPOCH, 'sparse_key_corridor')

from dissertation_files.agents.evaluation import plot_evaluation_data, plot_exploration_heatmap, load_file_for_plot
from dissertation_files.environments.minigrid_environments import RGBImgPartialObsWrapper
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

random_rewards = load_file_for_plot('sparse_multiroom', 'random', 'rewards', '2023-11-27')
dqn_rewards = load_file_for_plot('sparse_multiroom', 'dqn', 'rewards', '2023-11-27')
ppo_rewards = load_file_for_plot('sparse_multiroom', 'ppo', 'rewards', '2023-11-26')
rnd_rewards = load_file_for_plot('sparse_multiroom', 'rnd', 'rewards', '2023-11-25')

random_ftvs = load_file_for_plot('sparse_multiroom', 'random', 'ftvs', '2023-11-27')
dqn_ftvs = load_file_for_plot('sparse_multiroom', 'dqn', 'ftvs', '2023-11-27')
ppo_ftvs = load_file_for_plot('sparse_multiroom', 'ppo', 'ftvs', '2023-11-26')
rnd_ftvs = load_file_for_plot('sparse_multiroom', 'rnd', 'ftvs', '2023-11-25')

"""
Reward
"""

rewards = random_rewards | dqn_rewards | ppo_rewards | rnd_rewards
plot_evaluation_data(rewards, EPOCHS, EVALUATION_FREQUENCY, STEPS_PER_EPOCH, 'sparse_multiroom')

"""
Heatmaps
"""

plot_exploration_heatmap(env, random_ftvs, EPOCHS, STEPS_PER_EPOCH, 'sparse_multiroom')
plot_exploration_heatmap(env, dqn_ftvs, EPOCHS, STEPS_PER_EPOCH, 'sparse_multiroom')
plot_exploration_heatmap(env, ppo_ftvs, EPOCHS, STEPS_PER_EPOCH, 'sparse_multiroom')
plot_exploration_heatmap(env, rnd_ftvs, EPOCHS, STEPS_PER_EPOCH, 'sparse_multiroom')

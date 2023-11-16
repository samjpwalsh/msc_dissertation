from dissertation_files.agents.evaluation import plot_evaluation_data, plot_exploration_heatmap, \
    plot_state_visit_percentage, load_file_for_plot
from dissertation_files.environments.minigrid_environments import SparseSequentialRooms, FlatObsWrapper, RGBImgPartialObsWrapper
import sequential_rooms_test_pipeline
import gymnasium as gym

env = SparseSequentialRooms(render_mode='rgb_array')
env = RGBImgPartialObsWrapper(env)
env.reset()
EPOCHS = sequential_rooms_test_pipeline.EPOCHS
STEPS_PER_EPOCH = sequential_rooms_test_pipeline.STEPS_PER_EPOCH
EVALUATION_FREQUENCY = sequential_rooms_test_pipeline.EVALUATION_FREQUENCY

"""
Load data
"""

# random_rewards = load_file_for_plot('spiral_maze', 'random', 'rewards', '2023-10-01')
# dqn_rewards = load_file_for_plot('spiral_maze', 'dqn', 'rewards', '2023-10-01')
#ppo_rewards = load_file_for_plot('sparse_env_2', 'ppo', 'rewards', '2023-11-15')
rnd_rewards = load_file_for_plot('sparse_env_2', 'rnd', 'rewards', '2023-11-15')

# random_ftvs = load_file_for_plot('spiral_maze', 'random', 'ftvs', '2023-10-01')
# dqn_ftvs = load_file_for_plot('spiral_maze', 'dqn', 'ftvs', '2023-10-01')
#ppo_ftvs = load_file_for_plot('sparse_env_2', 'ppo', 'ftvs', '2023-11-15')
rnd_ftvs = load_file_for_plot('sparse_env_2', 'rnd', 'ftvs', '2023-11-15')

"""
Reward
"""

# rewards = random_rewards | dqn_rewards | ppo_rewards | rnd_rewards
# plot_evaluation_data(rewards, EPOCHS, EVALUATION_FREQUENCY, STEPS_PER_EPOCH, 'spiral_maze')

"""
State visit %
"""

# ftvs = random_ftvs | dqn_ftvs | ppo_ftvs | rnd_ftvs
# plot_state_visit_percentage(env, ftvs, EPOCHS, STEPS_PER_EPOCH, 'spiral_maze')

"""
Heatmaps
"""

# plot_exploration_heatmap(env, random_ftvs, EPOCHS, STEPS_PER_EPOCH, 'spiral_maze')
# plot_exploration_heatmap(env, dqn_ftvs, EPOCHS, STEPS_PER_EPOCH, 'spiral_maze')
#plot_exploration_heatmap(env, ppo_ftvs, EPOCHS, STEPS_PER_EPOCH, 'sparse_env_2')
plot_exploration_heatmap(env, rnd_ftvs, EPOCHS, STEPS_PER_EPOCH, 'sparse_env_2')
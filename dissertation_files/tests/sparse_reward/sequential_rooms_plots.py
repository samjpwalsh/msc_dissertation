from dissertation_files.agents.evaluation import plot_evaluation_data, plot_exploration_heatmap, load_file_for_plot
from dissertation_files.environments.minigrid_environments import SparseSequentialRooms, RGBImgPartialObsWrapper
import sequential_rooms_test_pipeline

env = SparseSequentialRooms(render_mode='rgb_array')
env = RGBImgPartialObsWrapper(env)
env.reset()
EPOCHS = sequential_rooms_test_pipeline.EPOCHS
STEPS_PER_EPOCH = sequential_rooms_test_pipeline.STEPS_PER_EPOCH
EVALUATION_FREQUENCY = sequential_rooms_test_pipeline.EVALUATION_FREQUENCY

"""
Load data
"""

random_rewards = load_file_for_plot('sparse_sequential_rooms', 'random', 'rewards', '2023-11-22')
dqn_rewards = load_file_for_plot('sparse_sequential_rooms', 'dqn', 'rewards', '2023-11-22')
ppo_rewards = load_file_for_plot('sparse_sequential_rooms', 'ppo', 'rewards', '2023-11-22')
rnd_rewards = load_file_for_plot('sparse_sequential_rooms', 'rnd', 'rewards', '2023-11-22')

random_ftvs = load_file_for_plot('sparse_sequential_rooms', 'random', 'ftvs', '2023-11-22')
dqn_ftvs = load_file_for_plot('sparse_sequential_rooms', 'dqn', 'ftvs', '2023-11-22')
ppo_ftvs = load_file_for_plot('sparse_sequential_rooms', 'ppo', 'ftvs', '2023-11-22')
rnd_ftvs = load_file_for_plot('sparse_sequential_rooms', 'rnd', 'ftvs', '2023-11-22')

"""
Reward
"""

rewards = random_rewards | dqn_rewards | ppo_rewards | rnd_rewards
plot_evaluation_data(rewards, EPOCHS, EVALUATION_FREQUENCY, STEPS_PER_EPOCH, 'sparse_sequential_rooms')

"""
Heatmaps
"""

plot_exploration_heatmap(env, random_ftvs, EPOCHS, STEPS_PER_EPOCH, 'sparse_sequential_rooms')
plot_exploration_heatmap(env, dqn_ftvs, EPOCHS, STEPS_PER_EPOCH, 'sparse_sequential_rooms')
plot_exploration_heatmap(env, ppo_ftvs, EPOCHS, STEPS_PER_EPOCH, 'sparse_sequential_rooms')
plot_exploration_heatmap(env, rnd_ftvs, EPOCHS, STEPS_PER_EPOCH, 'sparse_sequential_rooms')

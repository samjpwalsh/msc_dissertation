from dissertation_files.agents.evaluation import plot_evaluation_data, plot_exploration_heatmap, \
    plot_state_visit_percentage, load_file_for_plot
from dissertation_files.environments.minigrid_environments import FlatObsWrapper, TwoRooms
import two_rooms_test_pipeline

env = TwoRooms(render_mode=None)
env = FlatObsWrapper(env)
env.reset()
EPOCHS = two_rooms_test_pipeline.EPOCHS
STEPS_PER_EPOCH = two_rooms_test_pipeline.STEPS_PER_EPOCH
EVALUATION_FREQUENCY = two_rooms_test_pipeline.EVALUATION_FREQUENCY

"""
Load data
"""

random_rewards = load_file_for_plot('two_rooms', 'random', 'rewards', '2023-09-28')
dqn_rewards = load_file_for_plot('two_rooms', 'dqn', 'rewards', '2023-09-28')
ppo_rewards = load_file_for_plot('two_rooms', 'ppo', 'rewards', '2023-09-28')
rnd_rewards = load_file_for_plot('two_rooms', 'rnd', 'rewards', '2023-09-29')

random_ftvs = load_file_for_plot('two_rooms', 'random', 'ftvs', '2023-09-28')
dqn_ftvs = load_file_for_plot('two_rooms', 'dqn', 'ftvs', '2023-09-28')
ppo_ftvs = load_file_for_plot('two_rooms', 'ppo', 'ftvs', '2023-09-28')
rnd_ftvs = load_file_for_plot('two_rooms', 'rnd', 'ftvs', '2023-09-29')

"""
Reward
"""

rewards = random_rewards | dqn_rewards | ppo_rewards | rnd_rewards
plot_evaluation_data(rewards, EPOCHS, EVALUATION_FREQUENCY, STEPS_PER_EPOCH, 'two_rooms')

"""
State visit %
"""

ftvs = random_ftvs | dqn_ftvs | ppo_ftvs | rnd_ftvs
plot_state_visit_percentage(env, ftvs, EPOCHS, STEPS_PER_EPOCH, 'two_rooms')

"""
Heatmaps
"""

plot_exploration_heatmap(env, random_ftvs, EPOCHS, STEPS_PER_EPOCH, 'two_rooms')
plot_exploration_heatmap(env, dqn_ftvs, EPOCHS, STEPS_PER_EPOCH, 'two_rooms')
plot_exploration_heatmap(env, ppo_ftvs, EPOCHS, STEPS_PER_EPOCH, 'two_rooms')
plot_exploration_heatmap(env, rnd_ftvs, EPOCHS, STEPS_PER_EPOCH, 'two_rooms')



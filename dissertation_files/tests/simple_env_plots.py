from dissertation_files.agents.evaluation import plot_evaluation_data, plot_exploration_heatmap, \
    plot_state_visit_percentage, load_file_for_plot
from dissertation_files.environments.minigrid_environments import SimpleEnv
from dissertation_files.environments.minigrid_wrappers import FlatObsWrapper
import simple_env_test_pipeline

env = SimpleEnv(render_mode=None)
env = FlatObsWrapper(env)
env.reset()
EPOCHS = simple_env_test_pipeline.EPOCHS
STEPS_PER_EPOCH = simple_env_test_pipeline.STEPS_PER_EPOCH
EVALUATION_FREQUENCY = simple_env_test_pipeline.EVALUATION_FREQUENCY

"""
Load data
"""

random_rewards = load_file_for_plot('random', 'rewards', '2023-09-18')
dqn_rewards = load_file_for_plot('dqn', 'rewards', '2023-09-18')
ppo_rewards = load_file_for_plot('ppo', 'rewards', '2023-09-18')
rnd_rewards = load_file_for_plot('rnd', 'rewards', '2023-09-18')

random_ftvs = load_file_for_plot('random', 'ftvs', '2023-09-18')
dqn_ftvs = load_file_for_plot('dqn', 'ftvs', '2023-09-18')
ppo_ftvs = load_file_for_plot('ppo', 'ftvs', '2023-09-18')
rnd_ftvs = load_file_for_plot('rnd', 'ftvs', '2023-09-18')

"""
Heatmaps
"""

plot_exploration_heatmap(env, random_ftvs, EPOCHS, STEPS_PER_EPOCH, 'simple_env')
plot_exploration_heatmap(env, dqn_ftvs, EPOCHS, STEPS_PER_EPOCH, 'simple_env')
plot_exploration_heatmap(env, ppo_ftvs, EPOCHS, STEPS_PER_EPOCH, 'simple_env')
plot_exploration_heatmap(env, rnd_ftvs, EPOCHS, STEPS_PER_EPOCH, 'simple_env')

"""
Reward
"""

rewards = random_rewards | dqn_rewards | ppo_rewards | rnd_rewards
plot_evaluation_data(rewards, EPOCHS, EVALUATION_FREQUENCY, STEPS_PER_EPOCH, 'simple_env')

"""
State visit %
"""

ftvs = random_ftvs | dqn_ftvs | ppo_ftvs | rnd_ftvs
plot_state_visit_percentage(env, ftvs, EPOCHS, STEPS_PER_EPOCH, 'simple_env')

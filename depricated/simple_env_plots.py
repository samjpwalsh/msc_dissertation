from dissertation_files.agents.evaluation import plot_evaluation_data, plot_exploration_heatmap, \
    plot_state_visit_percentage, load_file_for_plot
from dissertation_files.environments.minigrid_environments import SimpleEnv, RGBImgPartialObsWrapper
import simple_env_test_pipeline

env = SimpleEnv(render_mode='rgb_array')
env = RGBImgPartialObsWrapper(env)
env.reset()
EPOCHS = simple_env_test_pipeline.EPOCHS
STEPS_PER_EPOCH = simple_env_test_pipeline.STEPS_PER_EPOCH
EVALUATION_FREQUENCY = simple_env_test_pipeline.EVALUATION_FREQUENCY

"""
Load data
"""

random_rewards = load_file_for_plot('simple_env', 'random', 'rewards', '2023-11-01')
dqn_rewards = load_file_for_plot('simple_env', 'dqn', 'rewards', '2023-11-01')
ppo_rewards = load_file_for_plot('simple_env', 'ppo', 'rewards', '2023-11-09')
rnd_rewards = load_file_for_plot('simple_env', 'rnd', 'rewards', '2023-11-02')

random_ftvs = load_file_for_plot('simple_env', 'random', 'ftvs', '2023-11-01')
dqn_ftvs = load_file_for_plot('simple_env', 'dqn', 'ftvs', '2023-11-01')
ppo_ftvs = load_file_for_plot('simple_env', 'ppo', 'ftvs', '2023-11-09')
rnd_ftvs = load_file_for_plot('simple_env', 'rnd', 'ftvs', '2023-11-02')

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

"""
Heatmaps
"""

plot_exploration_heatmap(env, random_ftvs, EPOCHS, STEPS_PER_EPOCH, 'simple_env')
plot_exploration_heatmap(env, dqn_ftvs, EPOCHS, STEPS_PER_EPOCH, 'simple_env')
plot_exploration_heatmap(env, ppo_ftvs, EPOCHS, STEPS_PER_EPOCH, 'simple_env')
plot_exploration_heatmap(env, rnd_ftvs, EPOCHS, STEPS_PER_EPOCH, 'simple_env')
from dissertation_files.agents.evaluation import plot_exploration_heatmap, load_file_for_plot
from dissertation_files.environments.minigrid_environments import DoubleSpiralMaze, RGBImgPartialObsWrapper
import double_spiral_test_pipeline

env = DoubleSpiralMaze(render_mode='rgb_array')
env = RGBImgPartialObsWrapper(env)
env.reset()
EPOCHS = double_spiral_test_pipeline.EPOCHS
STEPS_PER_EPOCH = double_spiral_test_pipeline.STEPS_PER_EPOCH

"""
Load data
"""

random_ftvs1 = load_file_for_plot('no_double_spiral', 'random', 'ftvs_2023-12-07', 'run_1')
random_ftvs2 = load_file_for_plot('no_double_spiral', 'random', 'ftvs_2023-12-07', 'run_2')
random_ftvs3 = load_file_for_plot('no_double_spiral', 'random', 'ftvs_2023-12-07', 'run_3')

dqn_ftvs1 = load_file_for_plot('no_double_spiral', 'dqn', 'ftvs_2023-12-07', 'run_1')
dqn_ftvs2 = load_file_for_plot('no_double_spiral', 'dqn', 'ftvs_2023-12-07', 'run_2')
dqn_ftvs3 = load_file_for_plot('no_double_spiral', 'dqn', 'ftvs_2023-12-07', 'run_3')

ppo_ftvs1 = load_file_for_plot('no_double_spiral', 'ppo', 'ftvs_2023-12-07', 'run_1')
ppo_ftvs2 = load_file_for_plot('no_double_spiral', 'ppo', 'ftvs_2023-12-07', 'run_2')
ppo_ftvs3 = load_file_for_plot('no_double_spiral', 'ppo', 'ftvs_2023-12-07', 'run_3')

rnd_ftvs1 = load_file_for_plot('no_double_spiral', 'rnd', 'ftvs_2023-12-06', 'run_1')
rnd_ftvs2 = load_file_for_plot('no_double_spiral', 'rnd', 'ftvs_2023-12-06', 'run_2')
rnd_ftvs3 = load_file_for_plot('no_double_spiral', 'rnd', 'ftvs_2023-12-06', 'run_3')
rnd_ftvs4 = load_file_for_plot('no_double_spiral', 'rnd', 'ftvs_2023-12-06', 'run_4')
rnd_ftvs5 = load_file_for_plot('no_double_spiral', 'rnd', 'ftvs_2023-12-06', 'run_5')
rnd_ftvs6 = load_file_for_plot('no_double_spiral', 'rnd', 'ftvs_2023-12-06', 'run_6')

"""
Heatmaps
"""

plot_exploration_heatmap(env, random_ftvs3, EPOCHS, STEPS_PER_EPOCH, 'no_double_spiral')
plot_exploration_heatmap(env, dqn_ftvs3, EPOCHS, STEPS_PER_EPOCH, 'no_double_spiral')
plot_exploration_heatmap(env, ppo_ftvs3, EPOCHS, STEPS_PER_EPOCH, 'no_double_spiral')
plot_exploration_heatmap(env, rnd_ftvs3, EPOCHS, STEPS_PER_EPOCH, 'no_double_spiral')

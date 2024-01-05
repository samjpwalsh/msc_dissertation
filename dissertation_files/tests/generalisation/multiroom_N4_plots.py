from dissertation_files.agents.evaluation import plot_evaluation_data, plot_exploration_heatmap, load_file_for_plot
from dissertation_files.environments.minigrid_environments import RGBImgPartialObsWrapper, MultiroomFourRooms
import multiroom_N4_pretraining

env = MultiroomFourRooms(render_mode='rgb_array')
env = RGBImgPartialObsWrapper(env)
env.reset()
EPOCHS = multiroom_N4_pretraining.EPOCHS
STEPS_PER_EPOCH = multiroom_N4_pretraining.STEPS_PER_EPOCH
EVALUATION_FREQUENCY = multiroom_N4_pretraining.ENVIRONMENT_INITIALISATIONS

"""
Load data
"""

rnd_rewards = load_file_for_plot('generalisation_multiroom_N4', 'rnd', 'rewards', '2023-12-21')
rnd_ftvs = load_file_for_plot('generalisation_multiroom_N4', 'rnd', 'ftvs', '2023-12-21')

"""
Reward
"""

plot_evaluation_data(rnd_rewards, EPOCHS, EVALUATION_FREQUENCY, STEPS_PER_EPOCH, 'generalisation_multiroom_N4')

"""
Heatmaps
"""

plot_exploration_heatmap(env, rnd_ftvs, EPOCHS, STEPS_PER_EPOCH, 'generalisation_multiroom_N4')

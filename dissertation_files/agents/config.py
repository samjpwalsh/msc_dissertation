from keras import activations

"""
Minigrid Hyperparameters
"""

# CNN

mg_conv_layers = [32, 64]
mg_kernel_size = [(8, 8), (4, 4)]
mg_strides = [(4, 4), (2, 2)]
mg_conv_hidden_activation = activations.relu

# DQN

mg_dqn_gamma = 0.95
mg_dqn_memory_size = 10000
mg_dqn_batch_size = 250
mg_dqn_hidden_sizes = (64, 64)
mg_dqn_input_activation = activations.relu
mg_dqn_output_activation = None
mg_dqn_learning_rate = 0.001
mg_dqn_epsilon = 1.0
mg_dqn_epsilon_decay = 0.999
mg_dqn_min_epsilon = 0.01
mg_dqn_steps_target_model_update = 100

# PPO

mg_ppo_gamma = 0.99
mg_ppo_clip_ratio = 0.2
mg_ppo_actor_learning_rate = 3e-4
mg_ppo_critic_learning_rate = 1e-3
mg_ppo_train_actor_iterations = 80
mg_ppo_train_critic_iterations = 80
mg_ppo_lam = 0.95
mg_ppo_hidden_sizes = (64, 64)
mg_ppo_input_activation = activations.relu
mg_ppo_output_activation = None

# RND

mg_rnd_hidden_sizes = (64, 64)
mg_rnd_hidden_sizes_generalisation = (128, 128, 128)
mg_rnd_input_activation = activations.relu
mg_rnd_output_activation = None
mg_rnd_actor_learning_rate = 3e-4
mg_rnd_critic_learning_rate = 1e-3
mg_rnd_rnd_predictor_learning_rate = 1e-4
mg_rnd_clip_ratio = 0.2
mg_rnd_gamma = 0.99
mg_rnd_lam = 0.95
mg_rnd_intrinsic_weight = 0.2
mg_rnd_train_actor_iterations = 80
mg_rnd_train_critic_iterations = 80
mg_rnd_train_rnd_iterations = 80

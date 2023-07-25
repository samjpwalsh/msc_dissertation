import numpy as np
from keras import activations
from dissertation_files.agents.utils import logprobabilities
from dissertation_files.agents.agent import RNDAgent
from dissertation_files.environments.simple_env import SimpleEnv
from dissertation_files.environments.minigrid_wrappers import FlatObsWrapper

"""
## Hyperparameters
"""
STEPS_PER_EPOCH = 4000
EPOCHS = 30
GAMMA = 0.99
CLIP_RATIO = 0.2
ACTOR_LEARNING_RATE = 3e-4
CRITIC_LEARNING_RATE = 1e-3
RND_PREDICTOR_LEARNING_RATE = 1e-3
TRAIN_ACTOR_ITERATIONS = 80
TRAIN_CRITIC_ITERATIONS = 80
TRAIN_RND_ITERATIONS = 80
LAM = 0.97
HIDDEN_SIZES = (64, 64)
INPUT_ACTIVATION = activations.relu
OUTPUT_ACTIVATION = None


"""
## Initializations
"""

env = SimpleEnv(render_mode=None)
env = FlatObsWrapper(env)
observation_dimensions = len(env.reset()[0])
action_dimensions = env.action_space.n

agent = RNDAgent(observation_dimensions, action_dimensions, STEPS_PER_EPOCH, HIDDEN_SIZES, INPUT_ACTIVATION,
                 OUTPUT_ACTIVATION, ACTOR_LEARNING_RATE, CRITIC_LEARNING_RATE, RND_PREDICTOR_LEARNING_RATE,
                 CLIP_RATIO, GAMMA, LAM)

observation, episode_return, episode_length = env.reset()[0], 0, 0
observation = np.reshape(observation, [1, observation_dimensions])

"""
## Training
"""

for epoch in range(EPOCHS):

    sum_return = 0
    sum_length = 0
    num_episodes = 0

    for t in range(STEPS_PER_EPOCH):

        logits, action = agent.sample_action(observation)
        observation_new, extrinsic_reward, done, truncated, _ = env.step(action[0].numpy())
        if truncated:
            done = True
        observation_new = np.reshape(observation_new, [1, observation_dimensions])
        intrinsic_reward = agent.calculate_intrinsic_reward(observation_new)
        episode_return += extrinsic_reward
        episode_length += 1

        value_t = agent.critic(observation)
        logprobability_t = logprobabilities(logits, action, action_dimensions)

        agent.buffer.store(observation, action, extrinsic_reward, intrinsic_reward, value_t, logprobability_t)

        observation = observation_new

        terminal = done
        if terminal or (t == STEPS_PER_EPOCH - 1):
            last_value = 0 if done else agent.critic(observation.reshape(1, -1))
            agent.buffer.finish_trajectory(last_value)
            sum_return += episode_return
            sum_length += episode_length
            num_episodes += 1
            observation, episode_return, episode_length = env.reset()[0], 0, 0
            observation = np.reshape(observation, [1, observation_dimensions])

    (
        observation_buffer,
        action_buffer,
        total_advantage_buffer,
        intrinsic_reward_buffer,
        total_reward_buffer,
        logprobability_buffer,
    ) = agent.buffer.get()

    for _ in range(TRAIN_ACTOR_ITERATIONS):
        agent.train_actor(observation_buffer, action_buffer, logprobability_buffer, total_advantage_buffer)

    for _ in range(TRAIN_CRITIC_ITERATIONS):
        agent.train_critic(observation_buffer, total_reward_buffer)
        # should the critic be trained on the total reward in RND or just extrinsic/intrinsic

    for _ in range(TRAIN_RND_ITERATIONS):
        agent.train_rnd_predictor(observation_buffer)

    print(
        f" Epoch: {epoch + 1}. Mean Return: {sum_return / num_episodes}. Mean Length: {sum_length / num_episodes}"
    )

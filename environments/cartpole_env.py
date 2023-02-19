import gymnasium as gym
import numpy as np

def cartpole(render=False):
    env = gym.make('CartPole-v1')
    # agent =
    # way of logging scores and outputting graph: score_logger =
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    episode = 0
    while True:
        episode += 1
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        step = 0
        while True:
            step += 1
            if render:
                env.render()
            action = agent.act(state)
            state_next, reward, terminal, info = env.step(action)
            reward = reward if not terminal else -reward
            state_next = np.reshape(state_next, [1, observation_space])
            agent.remember(state, action, reward, state_next, terminal)
            state = state_next
            if terminal:
                print "Run: " + str(episode) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step)
                score_logger.add_score(step, episode)
                break
            agent.experience_replay()

from dissertation_files.environments.simple_env import SimpleEnv
from dissertation_files.agents.agent import RandomAgent

env = SimpleEnv(render_mode="human")
action_dimensions = env.action_space.n
agent = RandomAgent(action_dimensions)
observation = env.reset()[0]
done = False
while not done:
    action = agent.sample_action(observation)
    next_observation, reward, done, truncated, _ = env.step(action)
    observation = next_observation
print(reward)

import os
import openai
import time
import gymnasium as gym

openai.api_key = os.environ['OPENAI_KEY']
model = "gpt-3.5-turbo"
temperature = 1
system_prompt = """
You are playing a game of cartpole, and are trying to maximise your cumulative reward.
All messages will be of the form [state, reward, done, cumulative reward].
=====================================================
The state: a four-tuple [Cart position, Cart velocity, Pole angle, Pole velocity]
Reward: 1 for each time step in the episode
Done: Boolean, True if the episode is over
Cumulative Reward: The sum of rewards received in the episode so far
=====================================================
Your action: You can only provide responses of '1', indicating you wish to push the cart to the right, or '0' to push the cart to the left.
=====================================================
Reminder, your response must be '1', or '0', no other text!!"""

env = gym.make('CartPole-v1')

episodes = 10

for i in range(episodes):
    state = env.reset()[0]
    reward = 0
    done = False
    cumulative_reward = 0
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": f"[{state}, {reward}, {done}, {cumulative_reward}]"}]
    while not done:
        for _ in range(10):
            try:
                chat = openai.ChatCompletion.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        stream=False)
                break
            except Exception as e:
                print(e)
                time.sleep(10)
        print(chat.choices[0].message.content)
        action = int(list(filter(str.isdigit, chat.choices[0].message.content))[0])
        state, reward, done, _, _ = env.step(action)
        cumulative_reward += reward
        messages[1] = {"role": "user", "content": f"[{state}, {reward}, {done}, {cumulative_reward}]"}
    print(f"Episode {i+1} Cumulative Reward = {cumulative_reward}")


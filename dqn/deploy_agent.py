"""  _
    |_|_
   _  | |
 _|_|_|_|_
|_|_|_|_|_|_
  |_|_|_|_|_|
    | | |_|
    |_|_
      |_|

Author: Souham Biswas
Website: https://www.linkedin.com/in/souham/
"""


import gym
import numpy as np

from dqn.nn import DQN


env_name = 'CartPole-v0'

# run_id = 0
# model_name = 'CartPole-v0--180.0--44835'
# model_fpath = os.sep.join(['atari_games', env_name, 'run-' + str(run_id), 'trained_models', model_name])

model_name = 'CartPole-v0'
model_fpath = 'final_models/' + model_name

epsilon = .05


if __name__ == '__main__':
    env = gym.make(env_name)
    curr_bgr_frame = env.reset()

    dqn = DQN(num_classes=env.action_space.n, model_prefix=env_name, optimized_inference=True,
              num_input_channels=1)
    dqn.load(model_fpath)
    curr_episode_reward = 0.
    all_episode_rewards = []

    while True:
        if np.random.rand() < epsilon:
            curr_action = env.action_space.sample()
        else:
            action_qvals = dqn.infer(np.expand_dims(np.expand_dims(curr_bgr_frame, 0), -1))
            curr_action = action_qvals.argmax()
        curr_bgr_frame, reward, done, info = env.step(curr_action)
        env.render()
        curr_episode_reward += reward
        if done:
            print(curr_episode_reward)
            curr_bgr_frame = env.reset()
            all_episode_rewards.append(curr_episode_reward)
            curr_episode_reward = 0.
    k = 0




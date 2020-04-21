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

from nn import DQN

GAMMA = .9


if __name__ == '__main__':
    env = gym.make("ChopperCommand-v0")

    dqn = DQN(num_classes=env.action_space.n)
    dqn.init()
    dqn.load()

    obs_new = env.reset()
    prev_num_lives = 3
    for step in range(10000000):
        env.render()
        obs_old = obs_new.copy()

        # random_action_prob = np.exp(-step / 10e3)
        random_action_prob = .5
        if np.random.rand() < random_action_prob:
            action = env.action_space.sample()
        else:
            action, _, _ = dqn.infer(obs_new)
            action = action[0]

        obs_new, reward, done, info = env.step(action)
        if info['ale.lives'] < prev_num_lives:
            reward = -10000.
        prev_num_lives = info['ale.lives']

        if reward != 0:
            train_prob = 1.
        else:
            train_prob = .03
            reward = -100
        reward = reward / 100.

        if np.random.rand() <= train_prob:
            _, _, action_qvals_net = dqn.infer(obs_new)
            y_gt = action_qvals_net.copy()
            # qval = y_gt[0, action]
            # y_gt[0, :] = 0.
            # y_gt[0, action] = qval + reward
            loss, step_tf, lr = dqn.train_step(obs_old, GAMMA * (y_gt + reward))
            print(step_tf, loss, random_action_prob, lr)
            if step_tf % 500 == 0:
                dqn.save()
        else:
            k = 0
            # print('Skipped train step...')
        if done or info['ale.lives'] == 0:
            observation = env.reset()
            prev_num_lives = 3
    env.close()




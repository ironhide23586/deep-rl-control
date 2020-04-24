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
import tensorflow as tf

from nn import DQN


BATCH_SIZE = 2


if __name__ == '__main__':
    env = gym.make("ChopperCommand-v0")

    dqn = DQN(num_classes=env.action_space.n)
    dqn.init()
    dqn.load()

    obs_new = env.reset()
    prev_num_lives = 3
    x_ims = []
    y_gts = []
    rewards = []
    actions = []

    experience_replay_buffer = []

    summary_writer = tf.summary.FileWriter('logs', dqn.sess.graph_def)

    conv_filters = [v for v in dqn.trainable_vars if 'conv' in v.name and 'bias' not in v.name]
    conv_summaries = []
    for filter_ in conv_filters:
        filter = tf.transpose(filter_, [3, 0, 1, 2])
        num_channels = 4
        num_viz = np.ceil(filter.shape[-1].value / num_channels).astype(np.int)
        splits = [num_channels] * num_viz
        splits[-1] = -1
        fvs = tf.split(filter, splits, axis=-1)
        for i in range(len(fvs)):
            conv_summaries.append(tf.summary.image(name=filter_.name + '/' + fvs[i].name, tensor=fvs[i]))
    random_action_prob = .5
    for step in range(10000000000):
        env.render()
        obs_old = obs_new.copy()

        random_action_prob = np.exp(-step / 10e4)
        # random_action_prob = 0.
        if np.random.rand() < random_action_prob:
            action = env.action_space.sample()
        else:
            action_pred, y_prob, y_pred = dqn.infer(obs_old)
            action = action_pred[0]
            # action = np.random.choice(np.arange(18), p=y_prob[0])

        obs_new, reward, done, info = env.step(action)

        if info['ale.lives'] < prev_num_lives:
            reward = -100.
        if reward > 0:
            reward *= .5
        prev_num_lives = info['ale.lives']

        if reward < 0:
            train_prob = 0.8
        elif reward > 0:
            train_prob = 1.
            # reward = 100.
        elif reward == 0:
            train_prob = 0.01
            reward = 0.
        reward = reward / 100.

        if np.random.rand() <= train_prob:
            an, _, action_qvals_net = dqn.infer(obs_new)
            y_gt = action_qvals_net.copy()
            # qval = y_gt.max()
            # y_gt[0, :] = 0.
            # y_gt[0, action] += reward
            x_ims.append(obs_old)
            y_gts.append(y_gt)
            rewards.append(reward)
            actions.append(action)
        else:
            k = 0
            # print('Skipped train step...')

        if len(x_ims) == BATCH_SIZE:
            x = np.array(x_ims)
            rewards = np.array(rewards)
            y = np.squeeze(np.array(y_gts))
            if BATCH_SIZE == 1:
                y = np.expand_dims(y, 0)
            experience_replay_buffer.append([x, y, rewards, actions])
            np.random.shuffle(experience_replay_buffer)
            x, y, rewards, actions = experience_replay_buffer[0]
            loss, step_tf, lr = dqn.train_step(x, y, rewards, actions)
            if len(experience_replay_buffer) > 150:
                experience_replay_buffer = experience_replay_buffer[-100:]

            for cs in conv_summaries:
                summary_writer.add_summary(cs.eval(session=dqn.sess), global_step=step_tf)

            print(step_tf, loss, random_action_prob, lr, rewards)
            if step_tf % 50 == 0:
                dqn.save(suffix=str(loss) + '-' + str(reward))
            x_ims = []
            y_gts = []
            rewards = []
            actions = []

        random_action_prob = np.clip(1. - reward, 0, 1.)

        if done or info['ale.lives'] == 0:
            observation = env.reset()
            prev_num_lives = 3
    env.close()




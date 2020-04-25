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
import cv2

from nn import DQN


class DQNEnvironment:

    def __init__(self, env_name="ChopperCommand-v0", num_lives=3, flicker_buffer_size=2,
                 sample_freq=4, replay_buffer_size=1000000, history_size=4, num_train_steps=1000000,
                 batch_size=32):
        self.env_name = env_name
        self.env = gym.make(self.env_name)
        self.num_lives = num_lives
        self.batch_size = batch_size
        self.dqn_final = DQN(num_classes=18)
        self.dqn_action = DQN(num_classes=18)
        self.step = 0
        self.flicker_buffer_size = flicker_buffer_size
        self.history_size = history_size
        self.sample_freq = sample_freq
        self.num_train_steps = num_train_steps
        self.replay_buffer_size = replay_buffer_size
        self.replay_buffer = []
        self.nn_input = []
        self.curr_frame = None
        self.curr_action = self.env.action_space.sample()
        self.init()

    def rewards_preprocess(self, reward, info):
        if info['ale.lives'] < self.num_lives:
            reward = -1.
        self.num_lives = info['ale.lives']
        reward = np.clip(reward, -1, +1)
        self.step += 1
        return reward

    def init(self):
        obs = self.env.reset()
        self.curr_frame = self.phi(obs)
        self.dqn_final.init()
        self.dqn_action.init()
        self.dqn_action.load()
        self.sync_params()
        for _ in range(self.history_size):
            self.curr_action = self.env.action_space.sample()
            self.perform_action(init_flag=True)
            self.nn_input.append(self.curr_frame)
        self.nn_input = np.expand_dims(np.rollaxis(np.array(self.nn_input), 0, 3), 0)
        self.populate_experience()

    def train_agent(self):
        for train_step in range(self.num_train_steps):
            self.train_step()
            for i in range(self.sample_freq):
                self.perform_action()
            self.populate_experience()
            if train_step % 50 == 0:
                self.sync_params()

    def populate_experience(self):
        self.random_action_prob = .9999977 ** self.step
        if np.random.rand() < self.random_action_prob:
              action = self.env.action_space.sample()
        else:
            action_pred, _, _ = self.dqn_final.infer(self.nn_input)
            action = action_pred[0]
        self.curr_action = action
        nn_input = self.nn_input.copy()
        reward = self.perform_action()
        experience = [nn_input[0], action, reward]
        actions_targ, action_probs, action_qvals = self.dqn_action.infer(self.nn_input)
        experience.append(action_qvals[0])
        if len(self.replay_buffer) == 0:
            self.replay_buffer = [experience]
        else:
            self.replay_buffer.append(experience)

    def phi(self, im):
        ims = [im]
        for _ in range(self.flicker_buffer_size - 1):
            im_new, reward, done, info = self.env.step(self.curr_action)
            ims.append(im_new)
        x = np.max(ims, axis=0)
        x = cv2.cvtColor(x, cv2.COLOR_RGB2YUV)[:, :, 0]
        x = cv2.resize(x, (84, 84))
        # cv2.imwrite('x.png', x)
        return x

    def perform_action(self, init_flag=False):
        obs, reward, done, info = self.env.step(self.curr_action)
        self.curr_frame = self.phi(obs)
        self.env.render()
        reward = self.rewards_preprocess(reward, info)
        if not init_flag:
            self.nn_input[0, :, :, :-1] = self.nn_input[0, :, :, 1:]
            self.nn_input[0, :, :, -1] = self.curr_frame
        return reward

    def sample_from_replay_memory(self):
        idx = (np.random.sample([min(len(self.replay_buffer), self.batch_size)])
               * len(self.replay_buffer)).astype(np.int)
        nn_inputs = []
        actions = []
        rewards = []
        y_targs = []
        for i in idx:
            nn_input, action, reward, y_targ = self.replay_buffer[i]
            nn_inputs.append(nn_input)
            actions.append(action)
            rewards.append(reward)
            y_targs.append(y_targ)
        nn_inputs = np.array(nn_inputs)
        actions = np.array(actions)
        rewards = np.array(rewards)
        y_targs = np.array(y_targs)
        return nn_inputs, actions, rewards, y_targs

    def train_step(self):
        if len(self.replay_buffer) > self.replay_buffer_size:
            self.replay_buffer = self.replay_buffer[-self.replay_buffer_size // 2:]
        nn_input, actions, rewards, y_targ = self.sample_from_replay_memory()
        loss, step_tf, lr, discount_factor = self.dqn_action.train_step(nn_input, y_targ, rewards, actions)
        print('Step =', step_tf, ', Loss=', loss, ', learn_rate =', lr, ', discount_factor =', discount_factor)

    def sync_params(self):
        print('Syncing Params of the 2 DQNs....')
        s = self.dqn_action.save()
        self.dqn_final.load(s)


if __name__ == '__main__':
    # dqn = DQN(num_classes=18)
    # dqn.init()
    # dqn.load()

    dqn_env = DQNEnvironment()
    dqn_env.train_agent()
    dqn_env.env.close()

    #
    # env = gym.make("ChopperCommand-v0")
    #
    # prev_num_lives = 3
    # x_ims = []
    # y_gts = []
    # rewards = []
    # actions = []
    # experience_replay_buffer = []
    #
    # summary_writer = tf.summary.FileWriter('logs', dqn.sess.graph_def)
    # conv_filters = [v for v in dqn.trainable_vars if 'conv' in v.name and 'bias' not in v.name]
    # conv_summaries = []
    # for filter_ in conv_filters:
    #     filter = tf.transpose(filter_, [3, 0, 1, 2])
    #     num_channels = 4
    #     num_viz = np.ceil(filter.shape[-1].value / num_channels).astype(np.int)
    #     splits = [num_channels] * num_viz
    #     splits[-1] = -1
    #     fvs = tf.split(filter, splits, axis=-1)
    #     for i in range(len(fvs)):
    #         conv_summaries.append(tf.summary.image(name=filter_.name + '/' + fvs[i].name, tensor=fvs[i]))
    #
    # s0 = env.reset()
    # action = env.action_space.sample()
    # s1, reward, done, info = env.step(action)
    #
    # for step in range(10000000000):
    #     env.render()
    #     obs_old = obs_new.copy()
    #
    #     random_action_prob = np.exp(-step / 10e4)
    #     # random_action_prob = 0.001
    #     if np.random.rand() < random_action_prob:
    #         action = env.action_space.sample()
    #     else:
    #         action_pred, y_prob, y_pred = dqn.infer(obs_old)
    #         action = action_pred[0]
    #     obs_new, reward, done, info = env.step(action)
    #
    #     if info['ale.lives'] < prev_num_lives:
    #         reward = -100.
    #     prev_num_lives = info['ale.lives']
    #
    #     reward = np.clip(reward, -1, +1)
    #
    #     an, _, action_qvals_net = dqn.infer(obs_new)
    #     y_gt = action_qvals_net.copy()
    #     x_ims.append(obs_old)
    #     y_gts.append(y_gt)
    #     rewards.append(reward)
    #     actions.append(action)
    #
    #     if len(x_ims) == BATCH_SIZE:
    #         x = np.array(x_ims)
    #         rewards = np.array(rewards)
    #         y = np.squeeze(np.array(y_gts))
    #         if BATCH_SIZE == 1:
    #             y = np.expand_dims(y, 0)
    #         experience_replay_buffer.append([x, y, rewards, actions])
    #         np.random.shuffle(experience_replay_buffer)
    #         x, y, rewards, actions = experience_replay_buffer[0]
    #         loss, step_tf, lr, gamma = dqn.train_step(x, y, rewards, actions)
    #         if len(experience_replay_buffer) > 150:
    #             experience_replay_buffer = experience_replay_buffer[-100:]
    #
    #         for cs in conv_summaries:
    #             summary_writer.add_summary(cs.eval(session=dqn.sess), global_step=step_tf)
    #
    #         print(step_tf, loss, random_action_prob, lr, rewards)
    #         if step_tf % 50 == 0:
    #             dqn.save(suffix=str(loss) + '-' + str(reward))
    #         x_ims = []
    #         y_gts = []
    #         rewards = []
    #         actions = []
    #
    #     random_action_prob = np.clip(1. - reward, 0, 1.)
    #
    #     if done or info['ale.lives'] == 0:
    #         observation = env.reset()
    #         prev_num_lives = 3
    # env.close()




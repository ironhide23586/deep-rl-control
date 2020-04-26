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

import time
from threading import Thread
from queue import Queue
import pickle
import os

import gym
import numpy as np
from tqdm import tqdm
import cv2
from matplotlib import pyplot as plt
from moviepy.editor import VideoClip

from nn import DQN


class DQNEnvironment:

    def __init__(self, env_name="ChopperCommand-v0", num_lives=3, flicker_buffer_size=2,
                 sample_freq=4, replay_buffer_size=1000000, history_size=4, num_train_steps=1000000,
                 batch_size=32, viz=True, sync_freq=10000, replay_start_size=0,
                 viz_duration_seconds=16000, viz_fps=60, episodic_reward_ema_alpha=.9,
                 nn_input_cache_fpath='nn_input.pkl', replay_memory_cache_fpath='replay_memory.pkl',
                 rewards_data_cache_fpath='rewards_history.pkl', loss_data_cache_fpath='loss_history.pkl'):
        self.env_name = env_name
        self.env = gym.make(self.env_name)
        self.num_lives = num_lives
        self.batch_size = batch_size
        self.sync_freq = sync_freq
        self.replay_start_size = replay_start_size
        self.replay_break_even_train_step = 0
        self.nn_input_cache_fpath = nn_input_cache_fpath
        self.replay_memory_cache_fpath = replay_memory_cache_fpath
        self.rewards_data_cache_fpath = rewards_data_cache_fpath
        self.loss_data_cache_fpath = loss_data_cache_fpath
        self.dqn_final = DQN(num_classes=18)
        self.dqn_action = DQN(num_classes=18)
        self.step = 0
        self.frame_count = 0
        self.curr_train_step = 0
        self.flicker_buffer_size = flicker_buffer_size
        self.history_size = history_size
        self.sample_freq = sample_freq
        self.num_train_steps = num_train_steps
        self.replay_buffer_size = replay_buffer_size
        self.random_action_prob = 1.
        self.replay_buffer = []
        self.nn_input = []
        self.plot_frame_indices = []
        self.curr_episode_rewards = []
        self.best_episode_rewards = []
        self.ema_episode_rewards = []
        self.plot_loss_train_steps = []
        self.plot_losses = []
        self.curr_frame = None
        self.curr_bgr_frame = None
        self.curr_action = self.env.action_space.sample()
        self.viz = viz
        self.curr_episode_reward = 0.
        self.best_episode_reward = 0.
        self.total_episode_ema_reward = 0.  # exponential moving average of all episodic rewards
        self.episodic_reward_ema_alpha = episodic_reward_ema_alpha
        if self.viz:
            self.video_out_fpath = 'chopper_command_outs/shm_dqn_chopper_command-' + str(time.time()) + '.mp4'
            self.video_buffer = Queue()
            self.viz_duration_seconds = viz_duration_seconds
            self.viz_fps = viz_fps
            self.video_writer_thread = Thread(target=self.video_writer)
            self.video_writer_thread.start()
        self.init()

    def video_writer(self):
        clip = VideoClip(self.make_frame, duration=3600*1000)
        clip.write_videofile(self.video_out_fpath, fps=self.viz_fps, verbose=False, logger=None)

    def make_frame(self, t):
        while self.video_buffer.empty():
            time.sleep(1)
        im, train_step, frame_count, curr_episode_reward, \
        best_episode_reward, total_episode_ema_reward, random_action_prob = self.video_buffer.get()
        cv2.putText(im, str(frame_count), (130, 10), cv2.FONT_HERSHEY_SIMPLEX, .3, (255, 255, 255))
        cv2.putText(im, str(train_step), (2, 10), cv2.FONT_HERSHEY_SIMPLEX, .3, (0, 0, 255))
        cv2.putText(im, 'EMA=' + str(round(total_episode_ema_reward, 3)), (50, 10), cv2.FONT_HERSHEY_SIMPLEX, .35,
                    (255, 255, 0))
        cv2.putText(im, 'Current=' + str(round(curr_episode_reward, 3)), (2, 205), cv2.FONT_HERSHEY_SIMPLEX, .35,
                    (255, 255, 0))
        cv2.putText(im, 'Best=' + str(round(best_episode_reward, 3)), (90, 205), cv2.FONT_HERSHEY_SIMPLEX, .35,
                    (0, 255, 0))
        cv2.putText(im, str(round(random_action_prob, 3)), (120, 180), cv2.FONT_HERSHEY_SIMPLEX, .35,
                    (13, 0, 0))
        return im

    def rewards_preprocess(self, reward, info):
        if info['ale.lives'] < self.num_lives:
            reward = -1.
        death = False
        if info['ale.lives'] < self.num_lives:
            death = True
        self.num_lives = info['ale.lives']
        reward = np.clip(reward, -1, +1)
        self.step += 1
        return reward, death

    def init(self):
        obs = self.env.reset()
        self.curr_frame = self.phi(obs)
        self.dqn_final.init()
        self.dqn_action.init()
        self.dqn_action.load()
        if not os.path.isfile(self.nn_input_cache_fpath):
            print(self.nn_input_cache_fpath, 'not found, initializing first nn_input with random actions')
            for _ in range(self.history_size):
                self.curr_action = self.env.action_space.sample()
                self.perform_action(init_flag=True)
                self.nn_input.append(self.curr_frame)
            self.nn_input = np.expand_dims(np.rollaxis(np.array(self.nn_input), 0, 3), 0)
        else:
            print(self.nn_input_cache_fpath, 'found!, reading from it....')
            self.nn_input = pickle.load(open(self.nn_input_cache_fpath, 'rb'))
        if not os.path.isfile(self.replay_memory_cache_fpath):
            print(self.replay_memory_cache_fpath, 'not found, building new experience replay...')
            self.populate_experience()
        else:
            print(self.replay_memory_cache_fpath, 'found!, reading from it...')
            self.replay_buffer = pickle.load(open(self.replay_memory_cache_fpath, 'rb'))
            self.replay_start_size = 0
        if os.path.isfile(self.rewards_data_cache_fpath):
            print(self.rewards_data_cache_fpath, 'found!, reading from it...')
            self.plot_frame_indices, self.curr_episode_rewards, \
            self.best_episode_rewards, self.ema_episode_rewards = pickle.load(open(self.rewards_data_cache_fpath, 'rb'))
            self.frame_count = self.plot_frame_indices[-1]
            self.best_episode_reward = self.best_episode_rewards[-1]
        if os.path.isfile(self.loss_data_cache_fpath):
            print(self.loss_data_cache_fpath, 'found!, reading from it...')
            self.plot_loss_train_steps, self.plot_losses = pickle.load(open(self.loss_data_cache_fpath, 'rb'))
        self.sync_and_save_params(init_mode=True)

    def train_agent(self):
        try:
            print('Warm-starting by collecting random experiences, NO TRAINING IS HAPPENING NOW!')
            print('Total progress-')
            while self.curr_train_step < self.num_train_steps:
                if self.frame_count >= self.replay_start_size:
                    self.train_step()
                for i in range(self.sample_freq):
                    self.perform_action()
                self.populate_experience()
                if self.frame_count >= self.replay_start_size:
                    self.random_action_prob = 1. - ((1. / self.num_train_steps) * self.curr_train_step)
                    self.random_action_prob = np.clip(self.random_action_prob, .1, 1.)
                if self.curr_train_step % self.sync_freq == 0 and self.curr_train_step > 0:
                    self.sync_and_save_params()
            self.clip.close()
            self.sync_and_save_params()
        except:
            if self.viz:
                self.clip.close()

    def plot_rewards(self):
        plt.plot(self.plot_frame_indices, self.curr_episode_rewards, '-', label='Current Episode')
        plt.plot(self.plot_frame_indices, self.best_episode_rewards, '-', label='Best Episode So Far')
        plt.plot(self.plot_frame_indices, self.ema_episode_rewards, '-', label='Exponential Moving Average')
        plt.legend(loc='best')
        plt.xlabel('Frame Count')
        plt.ylabel('Episodic Reward')
        plt.tight_layout()
        plt.savefig('rewards_plot.jpg')
        plt.clf()
        plt.plot(self.plot_loss_train_steps, self.plot_losses, '-', label='Loss')
        plt.legend(loc='best')
        plt.xlabel('Train Step')
        plt.ylabel('MSE Loss')
        plt.tight_layout()
        plt.savefig('loss_plot.jpg')
        plt.clf()

    def populate_experience(self):
        if self.frame_count < self.replay_start_size or np.random.rand() < self.random_action_prob:
              action = self.env.action_space.sample()
        else:
            action_pred, _, _ = self.dqn_action.infer(self.nn_input)
            action = action_pred[0]
        self.curr_action = action
        nn_input = self.nn_input.copy()
        reward, death = self.perform_action()
        experience = [nn_input[0], action, reward]
        actions_targ, action_probs, action_qvals = self.dqn_final.infer(self.nn_input)
        action_qvals = action_qvals[0]
        if death:
            action_qvals[actions_targ[0]] = 0.
        experience.append(action_qvals)
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
        return x

    def perform_action(self, init_flag=False):
        self.curr_bgr_frame, reward, done, info = self.env.step(self.curr_action)
        reward, death = self.rewards_preprocess(reward, info)
        self.curr_episode_reward += reward
        self.plot_frame_indices.append(int(self.frame_count))
        self.curr_episode_rewards.append(self.curr_episode_reward)
        self.best_episode_rewards.append(self.best_episode_reward)
        self.ema_episode_rewards.append(self.total_episode_ema_reward)
        self.frame_count += 1
        if self.frame_count % 4000 == 0:
            self.plot_rewards()
        if self.viz:
            self.video_buffer.put([self.curr_bgr_frame, self.curr_train_step, self.frame_count,
                                   self.curr_episode_reward, self.best_episode_reward, self.total_episode_ema_reward,
                                   self.random_action_prob])
        if self.frame_count == self.replay_start_size:
            self.replay_break_even_train_step = self.curr_train_step
            print('----+++------REACHED REPLAY BREAK-EVEN------+++----')
            print('Training the model now...')
        self.curr_frame = self.phi(self.curr_bgr_frame)
        # self.env.render()
        if not init_flag:
            self.nn_input[0, :, :, :-1] = self.nn_input[0, :, :, 1:]
            self.nn_input[0, :, :, -1] = self.curr_frame
        if done or self.num_lives <= 0:
            self.curr_bgr_frame = self.env.reset()
            self.num_lives = 3
            self.total_episode_ema_reward = self.total_episode_ema_reward + (self.episodic_reward_ema_alpha
                                              * (self.curr_episode_reward - self.total_episode_ema_reward))
            if self.curr_episode_reward > self.best_episode_reward:
                self.best_episode_reward = self.curr_episode_reward
            self.curr_episode_reward = 0.
        return reward, death

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
        self.curr_train_step = step_tf
        self.plot_loss_train_steps.append(self.curr_train_step)
        self.plot_losses.append(loss)
        if self.curr_train_step % 50 == 0:
            print('Step =', step_tf, ', Loss=', loss, ', random_action_prob =', self.random_action_prob)
            print('--> episodic_reward: EMA=', self.total_episode_ema_reward, ', BEST=', self.best_episode_reward, ', CURRENT=', self.curr_episode_reward)

    def sync_and_save_params(self, init_mode=False):
        print('Syncing Params of the 2 DQNs....')
        s = self.dqn_action.save(str(self.total_episode_ema_reward))
        self.dqn_final.load(s)
        if not init_mode:
            print('Writing', self.nn_input_cache_fpath)
            pickle.dump(self.nn_input, open(self.nn_input_cache_fpath, 'wb'))
            print('Writing', self.replay_memory_cache_fpath)
            pickle.dump(self.replay_buffer, open(self.replay_memory_cache_fpath, 'wb'))
            print('Writing', self.rewards_data_cache_fpath)
            pickle.dump([self.plot_frame_indices, self.curr_episode_rewards,
                         self.best_episode_rewards, self.ema_episode_rewards], open(self.rewards_data_cache_fpath, 'wb'))
            pickle.dump([self.plot_loss_train_steps, self.plot_losses], open(self.loss_data_cache_fpath, 'wb'))



if __name__ == '__main__':
    dqn_env = DQNEnvironment()
    dqn_env.train_agent()
    dqn_env.env.close()

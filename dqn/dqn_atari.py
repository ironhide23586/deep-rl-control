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
from glob import glob

import gym
import numpy as np
from tqdm import tqdm
import cv2
from matplotlib import pyplot as plt
from moviepy.editor import ImageSequenceClip

from nn import DQN


def force_makedir(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)


class DQNEnvironment:

    def __init__(self, env_name="Breakout-v0", root_dir='atari_games', flicker_buffer_size=2,
                 sample_freq=4, replay_buffer_size=1000000, history_size=4, num_train_steps=1000000,
                 batch_size=32, viz=True, sync_freq=10000, replay_start_size=50000, viz_fps=60,
                 episodic_reward_ema_alpha=.99, nn_input_cache_fname='nn_input', discount_factor=.99,
                 replay_memory_cache_fname='replay_memory', rewards_data_cache_fname='rewards_history',
                 loss_data_cache_fname='training_history', video_prefix='shm_dqn', run_dir_prefix='run',
                 print_loss_every_n_steps=100, render=True, plot_stride=100):
        self.env_name = env_name
        self.root_dir = root_dir
        self.viz_fps = viz_fps
        self.plot_stride = plot_stride
        self.render = render
        self.discount_factor = discount_factor
        self.print_loss_every_n_steps = print_loss_every_n_steps
        self.env_out_dir = self.root_dir + os.sep + self.env_name
        self.first_run = True
        force_makedir(self.env_out_dir)
        run_dirs = glob(self.env_out_dir + os.sep + '*')
        self.run_id = 0
        if len(run_dirs) > 0:
            run_indices = [int(s.split('-')[-1]) for s in run_dirs]
            self.run_id = max(run_indices) + 1
        self.prev_env_out_dir = self.env_out_dir + os.sep + run_dir_prefix + '-' + str(self.run_id - 1)
        self.curr_env_out_dir = self.env_out_dir + os.sep + run_dir_prefix + '-' + str(self.run_id)
        force_makedir(self.env_out_dir)
        self.curr_models_dir = self.curr_env_out_dir + os.sep + 'trained_models'
        self.prev_models_dir = self.prev_env_out_dir + os.sep + 'trained_models'
        force_makedir(self.curr_models_dir)
        self.curr_caches_dir = self.curr_env_out_dir + os.sep + 'caches'
        self.prev_caches_dir = self.prev_env_out_dir + os.sep + 'caches'
        force_makedir(self.curr_caches_dir)
        self.curr_plots_dir = self.curr_env_out_dir + os.sep + 'plots'
        self.prev_plots_dir = self.prev_env_out_dir + os.sep + 'plots'
        force_makedir(self.curr_plots_dir)
        if viz:
            self.video_out_dir = self.curr_env_out_dir + os.sep + 'video_outs'
            force_makedir(self.video_out_dir)
        self.env = gym.make(self.env_name)
        self.num_lives = None
        self.batch_size = batch_size
        self.sync_freq = sync_freq
        self.replay_start_size = replay_start_size
        self.replay_break_even_train_step = 0
        
        self.curr_nn_input_cache_fpath = self.curr_caches_dir + os.sep + nn_input_cache_fname
        self.curr_replay_memory_cache_fpath = self.curr_caches_dir + os.sep + replay_memory_cache_fname
        self.curr_rewards_data_cache_fpath = self.curr_caches_dir + os.sep + rewards_data_cache_fname
        self.curr_loss_data_cache_fpath = self.curr_caches_dir + os.sep + loss_data_cache_fname

        prev_suffix = ''
        if os.path.isdir(self.prev_caches_dir):
            prev_cache_nn_input_fpaths = glob(self.prev_caches_dir + os.sep + nn_input_cache_fname + '*')
            if len(prev_cache_nn_input_fpaths) > 0:
                latest_id = max([int(p.split('-')[-1]) for p in prev_cache_nn_input_fpaths])
                prev_suffix = str(latest_id)

        self.prev_nn_input_cache_fpath = self.prev_caches_dir + os.sep + nn_input_cache_fname + '-' + prev_suffix
        self.prev_replay_memory_cache_fpath = self.prev_caches_dir + os.sep + replay_memory_cache_fname \
                                              + '-' + prev_suffix
        self.prev_rewards_data_cache_fpath = self.prev_caches_dir + os.sep + rewards_data_cache_fname \
                                             + '-' + prev_suffix
        self.prev_loss_data_cache_fpath = self.prev_caches_dir + os.sep + loss_data_cache_fname + '-' + prev_suffix
        
        self.video_out_fpath_prefix = self.video_out_dir + os.sep + video_prefix + '-' + self.env_name + '-'
        self.dqn_constant = DQN(num_classes=self.env.action_space.n, model_folder=self.curr_models_dir,
                                model_prefix=self.env_name)
        self.dqn_action = DQN(num_classes=self.env.action_space.n, model_folder=self.curr_models_dir,
                              model_prefix=self.env_name)
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
        self.plot_loss_frame_counts = []
        self.plot_losses = []
        self.plot_l2_losses = []
        self.viz_frame_buffer = []
        self.viz_out_buffer = []
        self.curr_frame = None
        self.curr_bgr_frame = None
        self.prev_bgr_frame = None
        self.clip = None
        self.isalive = True
        self.curr_action = self.env.action_space.sample()
        self.viz = viz
        self.curr_episode_reward = 0.
        self.best_episode_reward = 0.
        self.total_episode_ema_reward = 0.  # exponential moving average of all episodic rewards
        self.episodic_reward_ema_alpha = episodic_reward_ema_alpha
        self.video_buffer = Queue()
        self.init()
        self.video_start_train_step = self.curr_train_step
        self.video_end_train_step = self.curr_train_step
        if self.viz:
            self.frame_writer_thread = Thread(target=self.make_frame)
            self.frame_writer_thread.start()

    def make_frame(self):
        while self.isalive:
            if self.video_buffer.empty():
                time.sleep(1)
                continue
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
            self.viz_frame_buffer.append(im)

    def rewards_preprocess(self, reward, info):
        death = False
        if info['ale.lives'] < self.num_lives:
            # reward = -1.
            death = True
        self.num_lives = info['ale.lives']
        reward = np.clip(reward, -1, +1)
        return reward, death

    def write_video_worker(self):
        if len(self.viz_out_buffer) > 0:
            video_out_fpath = self.video_out_fpath_prefix + str(self.video_start_train_step) + '-' \
                              + str(self.video_end_train_step) + '.mp4'
            self.video_start_train_step = self.video_end_train_step + 1
            clip = ImageSequenceClip(self.viz_out_buffer, fps=self.viz_fps)
            clip.write_videofile(video_out_fpath, fps=self.viz_fps, verbose=False, logger=None)
            self.viz_out_buffer = []

    def write_video(self):
        self.video_end_train_step = self.curr_train_step
        self.viz_out_buffer = [im.copy() for im in self.viz_frame_buffer]
        self.viz_frame_buffer = []
        video_writer_thread = Thread(target=self.write_video_worker)
        video_writer_thread.start()

    def init(self):
        self.curr_bgr_frame = self.env.reset()
        self.prev_bgr_frame = self.curr_bgr_frame.copy()
        self.phi()
        self.dqn_constant.init()
        self.dqn_action.init()
        self.dqn_action.load()
        if not os.path.isfile(self.prev_nn_input_cache_fpath):
            print(self.prev_nn_input_cache_fpath, 'not found, initializing first nn_input with random actions')
            for _ in range(self.history_size):
                self.curr_action = self.env.action_space.sample()
                self.perform_action(init_flag=True)
                self.nn_input.append(self.curr_frame)
            self.nn_input = np.expand_dims(np.rollaxis(np.array(self.nn_input), 0, 3), 0)
        else:
            print(self.prev_nn_input_cache_fpath, 'found!, reading from it....')
            self.nn_input = pickle.load(open(self.prev_nn_input_cache_fpath, 'rb'))
        if not os.path.isfile(self.prev_replay_memory_cache_fpath):
            print(self.prev_replay_memory_cache_fpath, 'not found, building new experience replay...')
            self.populate_experience()
        else:
            print(self.prev_replay_memory_cache_fpath, 'found!, reading from it...')
            self.replay_buffer = pickle.load(open(self.prev_replay_memory_cache_fpath, 'rb'))
        if os.path.isfile(self.prev_rewards_data_cache_fpath):
            print(self.prev_rewards_data_cache_fpath, 'found!, reading from it...')
            self.plot_frame_indices, self.curr_episode_rewards, \
            self.best_episode_rewards, self.ema_episode_rewards = pickle.load(open(self.prev_rewards_data_cache_fpath,
                                                                                   'rb'))
            self.frame_count = self.plot_frame_indices[-1]
            self.best_episode_reward = self.best_episode_rewards[-1]
            if self.frame_count > self.replay_start_size:
                self.print_train_start_text()
        if os.path.isfile(self.prev_loss_data_cache_fpath):
            print(self.prev_loss_data_cache_fpath, 'found!, reading from it...')
            self.plot_loss_frame_counts, self.plot_loss_train_steps, \
            self.plot_losses, self.plot_l2_losses = pickle.load(open(self.prev_loss_data_cache_fpath, 'rb'))
            self.curr_train_step = self.plot_loss_train_steps[-1]
        self.sync_and_save_params(init_mode=True)

    def train_agent(self):
        # try:
        print('Total progress-')
        if self.frame_count < self.replay_start_size:
            print('Warm-starting by collecting random experiences, NO TRAINING IS HAPPENING NOW!')
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
        self.die()
        # except:
        #     self.die()

    def print_train_start_text(self):
        print('----+++------REACHED REPLAY BREAK-EVEN at frame', self.frame_count, '------+++----')
        print('Training the model now as frame count is greater than specified replay size of', self.replay_start_size)

    def die(self):
        print('💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀-----> DEATH <-----💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀')
        self.isalive = False
        self.sync_and_save_params()
        self.dqn_action.sess.close()
        self.dqn_constant.sess.close()
        self.plot_stats()
        if self.viz:
            self.frame_writer_thread.join()
        print('Exiting....🏃')

    def plot_stats(self):
        idx = np.linspace(0, len(self.plot_frame_indices) - 1,
                          len(self.plot_frame_indices) // self.plot_stride).astype(np.int)
        frame_indices = np.array(self.plot_frame_indices)[idx]
        curr_r = np.array(self.curr_episode_rewards)[idx]
        best_r = np.array(self.best_episode_rewards)[idx]
        ema_r = np.array(self.ema_episode_rewards)[idx]
        plt.plot(frame_indices, curr_r, '-', label='Current Episode')
        plt.plot(frame_indices, best_r, '-', label='Best Episode So Far')
        plt.plot(frame_indices, ema_r, '-', label='Exponential Moving Average')
        plt.legend(loc='best')
        plt.xlabel('Frame Count')
        plt.ylabel('Episodic Reward')
        plt.tight_layout()
        plt.savefig(self.curr_plots_dir + os.sep + 'rewards_plot.jpg')
        plt.clf()
        idx = np.linspace(0, len(self.plot_loss_train_steps) - 1,
                          len(self.plot_loss_train_steps) // self.plot_stride).astype(np.int)
        loss_indices = np.array(self.plot_loss_train_steps)[idx]
        losses = np.clip(np.array(self.plot_losses)[idx], None, .1)
        l2_losses = np.clip(np.array(self.plot_l2_losses)[idx], None, .1)
        plt.plot(loss_indices, losses, '-', label='Loss')
        plt.plot(loss_indices, l2_losses, '-', label='L2 Loss')
        plt.legend(loc='best')
        plt.xlabel('Train Step')
        plt.ylabel('MSE Loss')
        plt.tight_layout()
        plt.savefig(self.curr_plots_dir + os.sep + 'loss_plot.jpg')
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
        actions_targ, action_probs, action_qvals = self.dqn_constant.infer(self.nn_input)
        max_qval = reward + self.discount_factor * action_qvals.max()
        if death:
            max_qval = reward
        experience = [nn_input[0], action, max_qval, max_qval]
        if len(self.replay_buffer) == 0:
            self.replay_buffer = [experience]
        else:
            self.replay_buffer.append(experience)

    def phi(self):
        ims = [self.prev_bgr_frame, self.curr_bgr_frame]
        x = np.max(ims, axis=0)
        x = cv2.cvtColor(x, cv2.COLOR_RGB2YUV)[:, :, 0]
        x = cv2.resize(x, (84, 84))
        self.curr_frame = x

    def perform_action(self, init_flag=False):
        self.prev_bgr_frame = self.curr_bgr_frame.copy()
        self.curr_bgr_frame, reward, done, info = self.env.step(self.curr_action)
        if self.first_run:
            self.first_run = False
            self.num_lives = info['ale.lives']
        self.phi()
        if self.render:
            self.env.render()
        reward, death = self.rewards_preprocess(reward, info)
        self.curr_episode_reward += reward
        self.plot_frame_indices.append(int(self.frame_count))
        self.curr_episode_rewards.append(self.curr_episode_reward)
        self.best_episode_rewards.append(self.best_episode_reward)
        self.ema_episode_rewards.append(self.total_episode_ema_reward)
        self.frame_count += 1
        if self.frame_count % 500 == 0:
            self.plot_stats()
        if self.viz:
            self.video_buffer.put([self.curr_bgr_frame.copy(), self.curr_train_step, self.frame_count,
                                   self.curr_episode_reward, self.best_episode_reward, self.total_episode_ema_reward,
                                   self.random_action_prob])
        if self.frame_count == self.replay_start_size:
            self.replay_break_even_train_step = self.curr_train_step
            self.print_train_start_text()
        if not init_flag:
            self.nn_input[0, :, :, :-1] = self.nn_input[0, :, :, 1:]
            self.nn_input[0, :, :, -1] = self.curr_frame
        if done or self.num_lives <= 0:
            self.curr_bgr_frame = self.env.reset()
            self.first_run = True
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
        l2_loss, loss, step_tf = self.dqn_action.train_step(nn_input, y_targ, rewards, actions)
        self.curr_train_step = step_tf
        self.plot_loss_frame_counts.append(self.frame_count)
        self.plot_loss_train_steps.append(self.curr_train_step)
        self.plot_l2_losses.append(l2_loss)
        self.plot_losses.append(loss)
        if self.curr_train_step % self.print_loss_every_n_steps == 0:
            print('Step =', step_tf, ', Loss=', loss, ', L2 Loss=', loss, ', random_action_prob =',
                  self.random_action_prob)
            print('--> episodic_reward: EMA=', self.total_episode_ema_reward, ', BEST=', self.best_episode_reward,
                  ', CURRENT=', self.curr_episode_reward)

    def sync_and_save_params(self, init_mode=False):
        print('Syncing Params of the 2 DQNs....')
        s = self.dqn_action.save(str(round(self.total_episode_ema_reward, 2)))
        self.dqn_constant.load(s)
        if not init_mode:
            suffix = '-' + str(self.curr_train_step)
            print('Writing', self.curr_nn_input_cache_fpath + suffix)
            pickle.dump(self.nn_input, open(self.curr_nn_input_cache_fpath + suffix, 'wb'))
            print('Writing', self.curr_replay_memory_cache_fpath + suffix)
            pickle.dump(self.replay_buffer, open(self.curr_replay_memory_cache_fpath + suffix, 'wb'))
            print('Writing', self.curr_rewards_data_cache_fpath + suffix)
            pickle.dump([self.plot_frame_indices, self.curr_episode_rewards,
                         self.best_episode_rewards, self.ema_episode_rewards],
                        open(self.curr_rewards_data_cache_fpath + suffix, 'wb'))
            pickle.dump([self.plot_loss_frame_counts, self.plot_loss_train_steps,
                         self.plot_losses, self.plot_l2_losses],
                        open(self.curr_loss_data_cache_fpath + suffix, 'wb'))
            self.write_video()
            k = 0


if __name__ == '__main__':
    dqn_env = DQNEnvironment()
    dqn_env.train_agent()
    dqn_env.env.close()

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
from shove import Shove

from nn import DQN


def force_makedir(dir):
    if not os.path.isdir(dir):
        print('Making directory at', dir)
        os.makedirs(dir)


def force_delete(fpath):
    if os.path.isfile(fpath):
        print('Deleting', fpath)
        os.remove(fpath)


class DQNEnvironment:

    def __init__(self, env_name="Breakout-v0", root_dir='atari_games', flicker_buffer_size=2,
                 sample_freq=4, replay_buffer_size=1000000, history_size=4, num_train_steps=1000000,
                 batch_size=32, viz=True, sync_freq=10000, replay_start_size=50000, viz_fps=60,
                 episodic_reward_ema_alpha=.99, discount_factor=.99, replay_memory_cache_fname='training_cache.db',
                 video_prefix='shm_dqn', run_dir_prefix='run', print_loss_every_n_steps=100, render=False,
                 plot_stride=100):
        self.env_name = env_name
        self.root_dir = root_dir
        self.viz_fps = viz_fps
        self.plot_stride = plot_stride
        self.render = render
        self.discount_factor = discount_factor
        self.print_loss_every_n_steps = print_loss_every_n_steps
        self.env_out_dir = self.root_dir + os.sep + self.env_name
        self.first_run = True
        self.experience_idx = 0
        self.l2_loss = 0.
        self.loss = 0.
        force_makedir(self.env_out_dir)
        run_dirs = glob(self.env_out_dir + os.sep + 'run-*')
        self.run_id = 0
        if len(run_dirs) > 0:
            run_indices = [int(s.split('-')[-1]) for s in run_dirs]
            self.run_id = max(run_indices) + 1
        self.cache_fpath = self.env_out_dir + os.sep + replay_memory_cache_fname
        self.curr_env_out_dir = self.env_out_dir + os.sep + run_dir_prefix + '-' + str(self.run_id)
        force_makedir(self.env_out_dir)
        self.curr_models_dir = self.curr_env_out_dir + os.sep + 'trained_models'
        force_makedir(self.curr_models_dir)
        self.curr_plots_dir = self.curr_env_out_dir + os.sep + 'plots'
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

        self.replay_buffer_db = None
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
        self.nn_input = np.zeros([1, self.dqn_action.im_h, self.dqn_action.im_w, self.history_size])
        self.plot_frame_indices = []
        self.curr_episode_rewards = []
        self.best_episode_rewards = []
        self.ema_episode_rewards = []
        self.plot_loss_train_steps = []
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
        self.random_action_taken = True
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
            best_episode_reward, total_episode_ema_reward, random_action_prob, \
            random_action_taken, curr_action, loss, l2_loss = self.video_buffer.get()
            dot_color = (0, 255, 0)
            if random_action_taken:
                dot_color = (0, 0, 255)
            cv2.putText(im, str(frame_count), (130, 10), cv2.FONT_HERSHEY_SIMPLEX, .3, (255, 255, 255))
            cv2.putText(im, str(train_step), (2, 10), cv2.FONT_HERSHEY_SIMPLEX, .3, (0, 0, 255))
            cv2.putText(im, 'EMA=' + str(round(total_episode_ema_reward, 3)), (50, 10), cv2.FONT_HERSHEY_SIMPLEX, .35,
                        (255, 255, 0))
            cv2.putText(im, 'Current=' + str(round(curr_episode_reward, 3)), (2, 205), cv2.FONT_HERSHEY_SIMPLEX, .35,
                        (255, 255, 0))
            cv2.putText(im, 'Best=' + str(round(best_episode_reward, 3)), (90, 205), cv2.FONT_HERSHEY_SIMPLEX, .35,
                        (0, 255, 0))
            cv2.putText(im, str(round(random_action_prob, 3)), (120, 180), cv2.FONT_HERSHEY_SIMPLEX, .35,
                        (255, 255, 255))
            cv2.circle(im, (5, 100), 5, dot_color, -1)
            cv2.putText(im, str(curr_action), (150, 100), cv2.FONT_HERSHEY_SIMPLEX, .35,
                        (0, 255, 255))
            cv2.putText(im, str(round(loss, 5)), (1, 110), cv2.FONT_HERSHEY_SIMPLEX, .2,
                        (255, 255, 255))
            cv2.putText(im, str(round(l2_loss, 5)), (1, 118), cv2.FONT_HERSHEY_SIMPLEX, .2,
                        (200, 200, 200))

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
        suffix = ''
        if self.frame_count <= self.replay_start_size:
            suffix = '-random_input'
        if len(self.viz_out_buffer) > 0:
            video_out_fpath = self.video_out_fpath_prefix + str(self.video_start_train_step) + '-' \
                              + str(self.video_end_train_step) + suffix + '.mp4'
            print('Writing video to', video_out_fpath)
            self.video_start_train_step = self.video_end_train_step + 1
            clip = ImageSequenceClip(self.viz_out_buffer, fps=self.viz_fps)
            clip.write_videofile(video_out_fpath, fps=self.viz_fps)  #, verbose=False, logger=None)
            self.viz_out_buffer = []

    def write_video(self):
        self.video_end_train_step = self.curr_train_step
        if self.frame_count <= self.replay_start_size:
            self.video_end_train_step = self.frame_count
        self.viz_out_buffer = [im.copy() for im in self.viz_frame_buffer]
        self.viz_frame_buffer = []
        video_writer_thread = Thread(target=self.write_video_worker)
        video_writer_thread.start()
        video_writer_thread.join()

    def init(self):
        self.curr_bgr_frame = self.env.reset()
        self.prev_bgr_frame = self.curr_bgr_frame.copy()
        self.phi()
        self.dqn_constant.init()
        self.dqn_action.init()
        self.dqn_action.load()
        self.curr_train_step = self.dqn_action.step
        db_exists = False
        if os.path.isfile(self.cache_fpath):
            db_exists = True
        self.replay_buffer_db = Shove('lite://' + self.cache_fpath)
        self.experience_idx = len(list(self.replay_buffer_db.keys()))
        if db_exists and self.experience_idx > 0:
            _, _, _, train_step_data = self.sample_from_replay_memory(random=False, chunk_size=1)
            cached_train_step, self.frame_count, self.total_episode_ema_reward, \
            self.curr_episode_reward, self.best_episode_reward = train_step_data
        for _ in range(self.history_size):
            self.curr_action = self.env.action_space.sample()
            self.random_action_taken = True
            self.perform_action()
        if not db_exists or self.experience_idx <= 0:
            self.populate_experience()
            self.sync_and_save_params(init_mode=True)

    def train_agent(self):
        try:
            print('Starting progress...')
            if self.frame_count < self.replay_start_size:
                print('Warm-starting by collecting random experiences, NO TRAINING IS HAPPENING NOW!')
            pbar = tqdm(total=self.num_train_steps)
            pbar.update(self.curr_train_step)
            while self.curr_train_step < self.num_train_steps:
                if self.frame_count >= self.replay_start_size:
                    self.train_step()
                    pbar.update(1)
                for i in range(self.sample_freq):
                    self.perform_action()
                self.populate_experience()
                if self.frame_count >= self.replay_start_size:
                    self.random_action_prob = 1. - ((1. / self.num_train_steps) * self.curr_train_step)
                    self.random_action_prob = np.clip(self.random_action_prob, .1, 1.)
                if self.curr_train_step % self.sync_freq == 0 and self.curr_train_step > 0:
                    self.sync_and_save_params()
            self.die()
        except:
            self.die()

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

    def plot_stats(self):  # TODO: accustom this to use ShoveDB
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
              self.random_action_taken = True
        else:
            action_pred, _, _ = self.dqn_action.infer(self.nn_input)
            self.random_action_taken = False
            action = action_pred[0]
        self.curr_action = action
        nn_input = self.nn_input.copy()
        reward, death = self.perform_action()
        actions_targ, action_probs, action_qvals = self.dqn_constant.infer(self.nn_input)
        max_qval = reward + self.discount_factor * action_qvals.max()
        if death:
            max_qval = reward
        experience = [nn_input[0], action, max_qval, self.curr_train_step, self.frame_count,
                      self.random_action_taken, self.total_episode_ema_reward, self.curr_episode_reward,
                      self.best_episode_reward]
        self.replay_buffer_db[self.experience_idx] = experience
        self.experience_idx += 1
        self.plot_frame_indices.append(self.frame_count)
        self.curr_episode_rewards.append(self.curr_episode_reward)
        self.best_episode_rewards.append(self.best_episode_reward)
        self.ema_episode_rewards.append(self.total_episode_ema_reward)

    def phi(self):
        ims = [self.prev_bgr_frame, self.curr_bgr_frame]
        x = np.max(ims, axis=0)
        x = cv2.cvtColor(x, cv2.COLOR_RGB2YUV)[:, :, 0]
        x = cv2.resize(x, (84, 84), interpolation=cv2.INTER_NEAREST)
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
        self.frame_count += 1
        if self.frame_count % self.print_loss_every_n_steps == 0:
            self.plot_stats()
        if self.viz:
            self.video_buffer.put([self.curr_bgr_frame.copy(), self.curr_train_step, self.frame_count,
                                   self.curr_episode_reward, self.best_episode_reward, self.total_episode_ema_reward,
                                   self.random_action_prob, self.random_action_taken, self.curr_action,
                                   self.loss, self.l2_loss])
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
        if self.frame_count % 36000 == 0:  # 10 mins of video
            self.write_video()
        return reward, death

    def sample_from_replay_memory(self, random=True, chunk_size=None):
        if chunk_size is None:
            chunk_size = self.batch_size
        if random:
            idx = (np.random.sample([min(len(self.replay_buffer_db.keys()), chunk_size)])
                   * len(self.replay_buffer_db.keys())).astype(np.int)
        else:
            start_idx = max(0, len(self.replay_buffer_db.keys()) - chunk_size)
            end_idx = min(start_idx + chunk_size, len(self.replay_buffer_db.keys()))
            idx = np.arange(start_idx, end_idx)
        nn_inputs = []
        actions = []
        max_qvals = []
        train_steps = []
        frame_counts = []
        ema_rs = []
        curr_rs = []
        best_rs = []
        for i in idx:
            nn_input, action, max_qval, train_step, frame_count, random_action_taken, ema_r, \
            curr_r, best_r = self.replay_buffer_db[int(i)]
            nn_inputs.append(nn_input)
            actions.append(action)
            max_qvals.append(max_qval)
            train_steps.append(train_step)
            frame_counts.append(frame_count)
            ema_rs.append(ema_r)
            curr_rs.append(curr_r)
            best_rs.append(best_r)
        nn_inputs = np.array(nn_inputs)
        actions = np.array(actions)
        max_qvals = np.array(max_qvals)

        i = np.argmax(train_steps)
        ts = train_steps[i]
        fc = frame_counts[i]
        er = ema_rs[i]
        cr = curr_rs[i]
        br = best_rs[i]

        train_data = [ts, fc, er, cr, br]  # [train_step, frame_count, ema_reward, curr_reward, best_reward]

        return nn_inputs, actions, max_qvals, train_data

    def train_step(self):
        nn_input, actions, y_targ, _ = self.sample_from_replay_memory(random=True, chunk_size=self.batch_size)
        self.l2_loss, self.loss, self.curr_train_step = self.dqn_action.train_step(nn_input, y_targ, actions)
        self.plot_loss_train_steps.append(self.curr_train_step)
        self.plot_losses.append(self.loss)
        self.plot_l2_losses.append(self.l2_loss)
        if self.curr_train_step % self.print_loss_every_n_steps == 0:
            print('Step =', self.curr_train_step, ', Loss=', self.loss, ', L2 Loss=', self.l2_loss,
                  ', random_action_prob =', self.random_action_prob)
            print('--> episodic_reward: EMA=', self.total_episode_ema_reward, ', BEST=', self.best_episode_reward,
                  ', CURRENT=', self.curr_episode_reward)

    def sync_and_save_params(self, init_mode=False):
        print('Syncing Params of the 2 DQNs....')
        s = self.dqn_action.save(str(round(self.total_episode_ema_reward, 2)))
        self.dqn_constant.load(s)
        if not init_mode:
            self.write_video()


if __name__ == '__main__':
    dqn_env = DQNEnvironment()
    dqn_env.train_agent()
    dqn_env.env.close()

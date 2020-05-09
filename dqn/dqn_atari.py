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
import os
from glob import glob
import shelve

import gym
import numpy as np
from tqdm.auto import tqdm
import cv2
from matplotlib import pyplot as plt
from moviepy.editor import ImageSequenceClip

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

    def __init__(self, env_name="Breakout-v0", root_dir='atari_games', flicker_buffer_size=2, sample_freq=4,
                 replay_buffer_size=1000000, history_size=4, num_train_steps=1000000,
                 batch_size=32, viz=True, sync_freq=10000, replay_start_size=50000, viz_fps=60, num_plot_points=100,
                 episodic_reward_ema_alpha=.7, discount_factor=.99, replay_memory_cache_fname='training_cache.db',
                 video_prefix='shm_dqn', run_dir_prefix='run', print_loss_every_n_steps=50, render=False,
                 max_replay_buffer_inmemory_size=50000, experience_db_sample_frac=.5,
                 refresh_replay_cache_every_n_experiences=900, write_video_every_n_frames=36000):
        self.env_name = env_name
        self.root_dir = root_dir
        self.viz_fps = viz_fps
        self.num_plot_points = num_plot_points
        self.write_video_every_n_frames = write_video_every_n_frames
        self.refresh_replay_cache_every_n_experiences = refresh_replay_cache_every_n_experiences
        self.max_replay_buffer_inmemory_size = max_replay_buffer_inmemory_size
        self.num_db_sampled_experiences = 0
        self.experience_db_sample_frac = experience_db_sample_frac
        self.n_db_sample = int(self.experience_db_sample_frac * self.max_replay_buffer_inmemory_size)
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
        self.replay_buffer_inmemory = []
        self.replay_buffer_db = None
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
        self.random_exploration_active = True
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
            cv2.putText(im, str(frame_count), (127, 10), cv2.FONT_HERSHEY_SIMPLEX, .3, (255, 255, 255))
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
            cv2.imwrite('im.png', im)
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
        print('Initializing DQN Environment...')
        self.curr_bgr_frame = self.env.reset()
        self.prev_bgr_frame = self.curr_bgr_frame.copy()
        self.phi()
        self.dqn_constant.init()
        self.dqn_action.init()
        self.dqn_action.load()
        self.curr_train_step = self.dqn_action.step
        self.random_action_prob = 1. - ((1. / self.num_train_steps) * self.curr_train_step)
        db_exists = os.path.isfile(self.cache_fpath + '.dat')
        if db_exists:
            print('Restoring train-state from training_cache at', self.cache_fpath + '.dat')
            self.replay_buffer_db = shelve.open(self.cache_fpath)
            db_keys = np.array(list(self.replay_buffer_db.keys()))
            db_train_steps = np.array([self.replay_buffer_db[k][5] for k in db_keys])
            db_frame_idx = np.array([self.replay_buffer_db[k][6] for k in db_keys])
            if self.curr_train_step == 0:
                last_experience_idx = db_keys[db_frame_idx[db_train_steps == 0].argmax()]
            else:
                last_experience_idx = db_keys[np.abs(db_train_steps - self.curr_train_step).argmin()]
            _, _, _, _, _, _, self.frame_count, _, self.total_episode_ema_reward, self.curr_episode_reward, \
            self.best_episode_reward, _, _ = self.replay_buffer_db[last_experience_idx]
            for dbk in db_keys:
                _, _, _, _, _, train_step, frame_count, _, total_episode_ema_reward, curr_episode_reward, \
                best_episode_reward, loss, l2_loss = self.replay_buffer_db[dbk]
                self.plot_frame_indices.append(frame_count)
                self.curr_episode_rewards.append(curr_episode_reward)
                self.best_episode_rewards.append(best_episode_reward)
                self.ema_episode_rewards.append(total_episode_ema_reward)
                self.plot_loss_train_steps.append(train_step)
                self.plot_losses.append(loss)
                self.plot_l2_losses.append(l2_loss)

            if self.frame_count >= self.replay_start_size:
                self.random_exploration_active = False
            self.replay_buffer_db.close()
            self.experience_idx = int(last_experience_idx) + 1
            self.write_and_refresh_replay_buffer_inmemory()
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
                pbar_frame_counts = tqdm(total=self.replay_start_size, position=0, leave=True)
                pbar_frame_counts.update(np.clip(self.frame_count, None, self.replay_start_size))
                pbar_train_steps = None
            else:
                pbar_train_steps = tqdm(total=self.num_train_steps, position=0, leave=True)
                pbar_train_steps.update(self.curr_train_step)
            prev_random_exploration_active = self.random_exploration_active
            prev_frame_count = self.frame_count
            while self.curr_train_step < self.num_train_steps:
                if self.frame_count >= self.replay_start_size:
                    self.train_step()
                    if pbar_train_steps is None:
                        pbar_train_steps = tqdm(total=self.num_train_steps, position=0, leave=True)
                    pbar_train_steps.update(1)
                    curr_random_exploration_active = self.random_exploration_active
                    if prev_random_exploration_active and not curr_random_exploration_active:
                        pbar_frame_counts.clear()
                else:
                    pbar_frame_counts.update(self.frame_count - prev_frame_count)
                    prev_frame_count = self.frame_count
                prev_random_exploration_active = self.random_exploration_active
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

    def plot_stats(self):
        idx = np.linspace(0, len(self.plot_frame_indices) - 1,
                          np.clip(self.num_plot_points, None,
                                  len(self.plot_frame_indices)).astype(np.int)).astype(np.int)
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
                          np.clip(self.num_plot_points, None,
                                  len(self.plot_loss_train_steps)).astype(np.int)).astype(np.int)
        loss_indices = np.array(self.plot_loss_train_steps)[idx]
        losses = np.clip(np.array(self.plot_losses)[idx], None, 1e-5)
        # l2_losses = np.clip(np.array(self.plot_l2_losses)[idx], None, .1)
        plt.plot(loss_indices, losses, '-', label='Loss')
        # plt.plot(loss_indices, l2_losses, '-', label='L2 Loss')
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
        nn_input_prev = self.nn_input.copy().astype(np.uint8)
        reward, death = self.perform_action()
        nn_input_new = self.nn_input.copy().astype(np.uint8)
        experience = [self.experience_idx, nn_input_prev[0], action, nn_input_new[0][:, :, -1], reward, death,
                      self.curr_train_step, self.frame_count, self.random_action_taken, self.total_episode_ema_reward,
                      self.curr_episode_reward, self.best_episode_reward, self.loss, self.l2_loss]
        self.replay_buffer_inmemory.append(experience)
        self.experience_idx += 1
        self.plot_frame_indices.append(self.frame_count)
        self.curr_episode_rewards.append(self.curr_episode_reward)
        self.best_episode_rewards.append(self.best_episode_reward)
        self.ema_episode_rewards.append(self.total_episode_ema_reward)
        self.plot_loss_train_steps.append(self.curr_train_step)
        self.plot_losses.append(self.loss)
        self.plot_l2_losses.append(self.l2_loss)
        if self.experience_idx % self.refresh_replay_cache_every_n_experiences == 0\
                or len(self.replay_buffer_inmemory) >= self.max_replay_buffer_inmemory_size:
            cache_writeout = not self.random_exploration_active
            self.write_and_refresh_replay_buffer_inmemory(refresh_inememory_experiences=cache_writeout)

    def write_and_refresh_replay_buffer_inmemory(self, refresh_inememory_experiences=True):
        if len(self.replay_buffer_inmemory) == 0:
            print('No experiences in memory, nothing to write to DB....')
            return
        self.replay_buffer_db = shelve.open(self.cache_fpath)
        record_len = len(self.replay_buffer_inmemory) - self.num_db_sampled_experiences
        print('Writing new', record_len, ' in-memory experiences to disk...')
        for i in tqdm(range(record_len), position=0, leave=True):
            exp = self.replay_buffer_inmemory[i + self.num_db_sampled_experiences]
            self.replay_buffer_db[str(exp[0])] = exp[1:]
        if self.random_exploration_active:
            self.replay_buffer_inmemory = []
        if refresh_inememory_experiences:
            self.replay_buffer_inmemory = []
            db_experience_indices = list(map(int, list(self.replay_buffer_db.keys())))
            print('Sampling random', self.n_db_sample, 'experiences from', len(db_experience_indices),
                  'experiences stored on disk and loading them to memory...')
            self.num_db_sampled_experiences = min(len(db_experience_indices), self.n_db_sample)
            selected_db_indices = sorted(map(int, np.random.choice(db_experience_indices,
                                                                   self.num_db_sampled_experiences)))
            for i in tqdm(range(len(selected_db_indices)), position=0, leave=True):
                experience = [selected_db_indices[i]] + self.replay_buffer_db[str(selected_db_indices[i])]
                self.replay_buffer_inmemory.append(experience)
        self.replay_buffer_db.close()

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
            self.random_exploration_active = False
            self.print_train_start_text()
        if not init_flag:
            self.nn_input[0, :, :, :-1] = self.nn_input[0, :, :, 1:]
            self.nn_input[0, :, :, -1] = self.curr_frame
        if done or self.num_lives <= 0:
            self.curr_bgr_frame = self.env.reset()
            self.first_run = True
            self.total_episode_ema_reward = (1 - self.episodic_reward_ema_alpha) * self.total_episode_ema_reward + \
                                            self.episodic_reward_ema_alpha * (self.curr_episode_reward -
                                                                              self.total_episode_ema_reward)
            if self.curr_episode_reward > self.best_episode_reward:
                self.best_episode_reward = self.curr_episode_reward
            self.curr_episode_reward = 0.
        if self.frame_count % self.write_video_every_n_frames == 0:
            self.write_video()
        return reward, death

    def sample_from_replay_memory(self, random=True, chunk_size=None):
        while len(self.replay_buffer_inmemory) <= 0:
            print('Waiting for in-memory replay buffer to be populated....')
            time.sleep(1)
        if chunk_size is None:
            chunk_size = self.batch_size
        if random:
            idx = (np.random.sample([min(len(self.replay_buffer_inmemory), chunk_size)])
                   * len(self.replay_buffer_inmemory)).astype(np.int)
        else:
            start_idx = max(0, len(self.replay_buffer_inmemory) - chunk_size)
            end_idx = min(start_idx + chunk_size, len(self.replay_buffer_inmemory))
            idx = np.arange(start_idx, end_idx)
        nn_inputs_prev = []
        actions = []
        y_targs = []
        train_steps = []
        frame_counts = []
        ema_rs = []
        curr_rs = []
        best_rs = []
        for i in idx:
            _, nn_input_prev, action, new_frame, reward, death, train_step, frame_count, random_action_taken,\
            ema_r, curr_r, best_r, _, _ = self.replay_buffer_inmemory[int(i)]
            nn_input_new = np.zeros_like(nn_input_prev)
            nn_input_new[:, :, :-1] = nn_input_prev[:, :, 1:]
            nn_input_new[:, :, -1] = new_frame
            actions_targ, action_probs, action_qvals = self.dqn_constant.infer(np.expand_dims(nn_input_new, 0))
            max_qval = reward + self.discount_factor * action_qvals.max()
            if death:
                max_qval = reward
            nn_inputs_prev.append(nn_input_prev)
            actions.append(action)
            y_targs.append(max_qval)

            train_steps.append(train_step)
            frame_counts.append(frame_count)
            ema_rs.append(ema_r)
            curr_rs.append(curr_r)
            best_rs.append(best_r)

        nn_inputs_prev = np.array(nn_inputs_prev)
        actions = np.array(actions)

        i = np.argmax(train_steps)
        ts = train_steps[i]
        fc = frame_counts[i]
        er = ema_rs[i]
        cr = curr_rs[i]
        br = best_rs[i]

        train_data = [ts, fc, er, cr, br]  # [train_step, frame_count, ema_reward, curr_reward, best_reward]

        return nn_inputs_prev, actions, y_targs, train_data

    def train_step(self):
        # sample method also obtains new q vals by running dqn_constant on it
        nn_input, actions, y_targs, _ = self.sample_from_replay_memory(random=True, chunk_size=self.batch_size)
        self.l2_loss, self.loss, self.curr_train_step = self.dqn_action.train_step(nn_input, y_targs, actions)
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
        self.write_and_refresh_replay_buffer_inmemory(refresh_inememory_experiences=False)


if __name__ == '__main__':
    dqn_env = DQNEnvironment()
    dqn_env.train_agent()

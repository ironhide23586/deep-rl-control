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

import os
from glob import glob

import tensorflow as tf
import numpy as np
import cv2


class DQN:

    def __init__(self, num_classes, im_w=160, im_h=210, compute_bn_mean_var=False, start_step=0, dropout_enabled=False,
                 learn_rate=5e-4, l2_regularizer_coeff=1e-2, num_steps=10000, dropout_rate=.3,
                 update_batchnorm_means_vars=False, optimized_inference=False, load_training_vars=False):
        self.model_folder = 'all_trained_models/trained_models'
        if not os.path.isdir(self.model_folder):
            os.makedirs(self.model_folder)
        self.model_fpath_prefix = self.model_folder + '/' + 'dqn_atari_chopper_command-'
        self.num_classes = num_classes
        self.im_h = im_h
        self.im_w = im_w
        self.compute_bn_mean_var = compute_bn_mean_var
        self.optimized_inference = optimized_inference
        self.x_tensor = tf.placeholder(tf.float32, shape=[None, self.im_h, self.im_w, 3],
                                       name='input_x_tensor')
        self.rewards_tensor = tf.placeholder(tf.float32, shape=[None], name='rewards_tensor')
        self.actions_tensor = tf.placeholder(tf.int32, shape=[None], name='actions_tensor')
        self.layers = [self.x_tensor]

        self.start_step = start_step
        self.step = start_step
        self.learn_rate = learn_rate
        self.step_ph = tf.Variable(self.start_step, trainable=False, name='train_step')
        self.learn_rate_tf = tf.train.exponential_decay(self.learn_rate, self.step_ph, num_steps, decay_rate=0.068,
                                                        name='learn_rate')
        self.unsaved_vars = [self.step_ph, self.learn_rate_tf]

        self.sess = None
        if self.optimized_inference:
            self.dropout_enabled = False
            self.out_op, _, _, _ = self.init_nn_graph()
            self.outs_softmax_op = tf.nn.softmax(self.out_op)
            self.outs_final = tf.argmax(self.outs_softmax_op, axis=-1), self.outs_softmax_op, self.out_op
            self.vars_to_keep = [v for v in tf.global_variables() if v not in self.unsaved_vars]
            self.restorer = tf.train.Saver(var_list=self.vars_to_keep)
            return
        self.dropout_enabled = dropout_enabled
        self.l2_regularizer_coeff = l2_regularizer_coeff
        self.y_tensor = tf.placeholder(tf.float32, shape=(None, self.num_classes), name='input_y_qvals')
        if self.dropout_enabled:
            self.dropout_rate = dropout_rate
            self.dropout_rate_tensor = tf.placeholder(tf.float32, shape=())
        self.out_op, self.trainable_vars, self.restore_excluded_vars, stop_grad_vars_and_ops = self.init_nn_graph()
        self.outs_softmax_op = tf.nn.softmax(self.out_op)
        self.actions = tf.argmax(self.out_op, axis=-1)
        self.outs_final = self.actions, self.outs_softmax_op, self.out_op

        self.stop_grad_vars, self.stop_grad_update_ops = stop_grad_vars_and_ops

        # self.loss_op = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_tensor,
        #                                                               logits=self.out_op)

        # args = tf.transpose(tf.stack([self.rewards_tensor, tf.cast(self.actions, tf.float32)]))
        # args = tf.reshape(tf.concat([self.out_op, args], axis=1), [-1, 1, -1])
        #
        # a = tf.map_fn(self.update_fn, args, dtype=tf.float32)

        self.rewards_mask = tf.one_hot(self.actions_tensor, self.num_classes) \
                            * .5 * tf.transpose(tf.tile([self.rewards_tensor], [self.num_classes, 1]))
        self.y_pred = self.out_op + self.rewards_mask

        self.loss_op = tf.reduce_mean(tf.math.squared_difference(self.y_tensor, self.y_pred))

        # self.loss_op = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y_tensor, logits=self.out_op)

        l2_losses = [self.l2_regularizer_coeff * tf.nn.l2_loss(v) for v in self.trainable_vars]
        self.reduced_loss = tf.reduce_mean(self.loss_op) + tf.add_n(l2_losses)

        # self.opt = tf.train.AdamOptimizer(learning_rate=self.learn_rate_tf)
        self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.learn_rate_tf)
        grads = tf.gradients(self.reduced_loss, self.trainable_vars, stop_gradients=self.stop_grad_vars)

        if update_batchnorm_means_vars:
            update_ops_all = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            update_ops = [op for op in update_ops_all if op not in self.stop_grad_update_ops]
            with tf.control_dependencies(update_ops):
                self.train_op = self.opt.apply_gradients(zip(grads, self.trainable_vars), global_step=self.step_ph)
        else:
            self.train_op = self.opt.apply_gradients(zip(grads, self.trainable_vars), global_step=self.step_ph)

        if not load_training_vars:
            self.restore_excluded_vars += [v for v in tf.all_variables() if 'Adam' in v.name or 'power' in v.name]
        else:
            self.restore_excluded_vars += []

        self.vars_to_keep = [v for v in tf.global_variables() if v not in self.unsaved_vars]
        self.vars_to_restore = [v for v in self.vars_to_keep if v not in self.restore_excluded_vars]

        self.saver = tf.train.Saver(max_to_keep=0, var_list=self.vars_to_keep)
        self.restorer = tf.train.Saver(var_list=self.vars_to_restore)

    def init(self):
        if not self.sess:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            init = tf.global_variables_initializer()
            self.sess.run(init)

    # def update_fn(self, x_in):
    #     x = x_in[0]
    #     q = x[:self.num_classes]
    #     # reward, action = x[self.num_classes:]
    #     action = tf.cast(x[-1], tf.int32)
    #     tf.one_hot()
    #     # q = tf.scatter_nd_add(q, action, x[-2])
    #     q[action] = q[action] + x[-2]
    #     return q

    def save(self, suffix=None):
        if self.optimized_inference:
            self.restorer.save(self.sess, 'dqn_atari_chopper_command')
            print('Model Saved in optimized inference mode')
            return
        if suffix:
            save_fpath = self.model_fpath_prefix + '-' + suffix + '--' + str(self.step)
        else:
            save_fpath = self.model_fpath_prefix + '-' + str(self.step)
        self.saver.save(self.sess, save_fpath)
        print('Model saved at', save_fpath)

    def load(self, model_path=None):
        if not self.sess:
            self.init()
        if model_path is None:
            if os.path.isdir(self.model_folder):
                existing_paths = glob(self.model_folder + '/*.index')
                if len(existing_paths) == 0:
                    print('No model found to restore from, initializing random weights')
                    return
                existing_ids = [int(p.split('--')[-1].replace('.index', '')) for p in existing_paths]
                selected_idx = np.argmax(existing_ids)
                self.step = existing_ids[selected_idx]
                self.start_step = self.step
                model_path = existing_paths[selected_idx].replace('.index', '')
            else:
                print('No model found to restore from, initializing random weights')
                return
        self.restorer.restore(self.sess, model_path)
        if not self.optimized_inference:
            step_assign_op = tf.assign(self.step_ph, self.start_step)
            self.sess.run(step_assign_op)
        print('Model restored from', model_path)

    def infer(self, im_in):
        if len(im_in.shape) > 3:
            im = ((im_in[:, :, :, [2, 1, 0]] / 255.) * 2) - 1
        else:
            im = np.expand_dims(((im_in[:, :, [2, 1, 0]] / 255.) * 2) - 1, 0)
        if self.dropout_enabled:
            outs = self.sess.run(self.outs_final, feed_dict={self.x_tensor: im,
                                                             self.dropout_rate_tensor: 0.})
        else:
            outs = self.sess.run(self.outs_final, feed_dict={self.x_tensor: im})
        return outs  # TODO: somehting wrong with final conv layer output; convolution outputting all zeros

    def center_crop(self, x):
        h, w, _ = x.shape
        offset = abs((w - h) // 2)
        if h < w:
            x_pp = x[:, offset:offset + h, :]
        elif w < h:
            x_pp = x[offset:offset + w, :, :]
        else:
            x_pp = x.copy()
        return x_pp

    def infer_optimized(self, im_in):
        im = self.center_crop(im_in)
        h, w, _ = im.shape
        if h != self.im_h or w != self.im_w:
            im = cv2.resize(im, (self.im_w, self.im_h))
        im = ((im[:, :, [2, 1, 0]] / 255.) * 2) - 1
        im = np.expand_dims(im, 0)
        out_label_idx, out_label_conf, out_label_logits = self.sess.run(self.outs_final, feed_dict={self.x_tensor: im})
        return out_label_idx, out_label_conf, out_label_logits

    def train_step(self, x_in, y, rewards, actions):
        if len(x_in.shape) == 3:
            x = np.expand_dims(x_in, axis=0)
        else:
            x = x_in.copy()
        x = ((x[:, :, :, [2, 1, 0]] / 255.) * 2) - 1
        if self.dropout_enabled:
            loss, _, step_tf, lr = self.sess.run([self.reduced_loss, self.train_op, self.step_ph, self.learn_rate_tf],
                                                 feed_dict={self.x_tensor: x,
                                                            self.y_tensor: y,
                                                            self.rewards_tensor: rewards,
                                                            self.actions_tensor: actions,
                                                            self.dropout_rate_tensor: self.dropout_rate})
        else:
            loss, _, step_tf, lr = self.sess.run([self.reduced_loss, self.train_op, self.step_ph, self.learn_rate_tf],
                                                 feed_dict={self.x_tensor: x,
                                                            self.y_tensor: y,
                                                            self.rewards_tensor: rewards,
                                                            self.actions_tensor: actions})
        self.step = step_tf
        return loss, step_tf, lr

    def conv_block(self, x_in, output_filters, kernel_size=3, kernel_stride=1, dilation=1, padding="VALID",
                   batch_norm=True, activation=tf.nn.tanh, pooling=True, pool_ksize=3, pool_stride=1,
                   pool_padding="VALID", pooling_fn=tf.nn.max_pool, block_depth=1, make_residual=True,
                   compute_bn_mean_var=None):
        if compute_bn_mean_var is None:
            compute_bn_mean_var = self.compute_bn_mean_var
        if not batch_norm:
            use_bias = True
        else:
            use_bias = False
        curr_layer = []
        layer_out = x_in
        if block_depth == 1:
            make_residual = False
        for depth in range(block_depth):
            layer_out = tf.layers.conv2d(layer_out, output_filters, kernel_size, strides=kernel_stride,
                                         use_bias=use_bias, activation=activation, dilation_rate=dilation,
                                         padding=padding)
            curr_layer.append(layer_out)
            if pooling:
                layer_out = pooling_fn(layer_out, ksize=[1, pool_ksize, pool_ksize, 1],
                                       strides=[1, pool_stride, pool_stride, 1], padding=pool_padding)
                curr_layer.append(layer_out)
            if batch_norm:
                layer_out = tf.layers.batch_normalization(layer_out, training=compute_bn_mean_var)
                curr_layer.append(layer_out)
            if depth == 0:
                residual_input = layer_out
            output = layer_out
        if make_residual:
            output = output + tf.image.resize_bilinear(residual_input, output.shape[1:3])
            curr_layer.append(output)
            if batch_norm:
                output = tf.layers.batch_normalization(output, training=compute_bn_mean_var)
                curr_layer.append(output)
        if self.dropout_enabled:
            output = tf.nn.dropout(output, rate=self.dropout_rate_tensor)
            curr_layer.append(output)
        self.layers.append(curr_layer)
        return output

    def dense_block(self, x_in, num_outs, batch_norm=True, biased=False, activation=tf.nn.relu6):
        curr_layer = []
        layer_outs = tf.layers.dense(x_in, num_outs, use_bias=biased)
        curr_layer.append(layer_outs)
        layer_outs = activation(layer_outs)
        curr_layer.append(layer_outs)
        if batch_norm:
            layer_outs = tf.layers.batch_normalization(layer_outs, training=self.compute_bn_mean_var, axis=0)
            curr_layer.append(layer_outs)
        if self.dropout_enabled:
            layer_outs = tf.nn.dropout(layer_outs, rate=self.dropout_rate_tensor)
            curr_layer.append(layer_outs)
        self.layers.append(curr_layer)
        return layer_outs

    def __get_nn_vars_and_ops(self):
        vars = tf.global_variables()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        return vars, update_ops

    def init_nn_graph(self):
        feature_extractor_bn_mean_var_compute = False
        layer_outs = self.conv_block(self.x_tensor, 4, compute_bn_mean_var=feature_extractor_bn_mean_var_compute)
        layer_outs = self.conv_block(layer_outs, 16, pool_ksize=4, pool_stride=1, block_depth=2,
                                     compute_bn_mean_var=feature_extractor_bn_mean_var_compute)
        layer_outs = self.conv_block(layer_outs, 128, pool_ksize=4, pool_stride=2, block_depth=1,
                                     compute_bn_mean_var=feature_extractor_bn_mean_var_compute)
        layer_outs = self.conv_block(layer_outs, 256, pooling=False,
                                     compute_bn_mean_var=feature_extractor_bn_mean_var_compute)
        layer_outs = self.conv_block(layer_outs, 32, pool_ksize=4, pool_stride=2, block_depth=2,
                                     compute_bn_mean_var=feature_extractor_bn_mean_var_compute)
        # stop_grad_vars_and_ops = self.__get_nn_vars_and_ops()
        # shp = layer_outs.shape
        # flat_len = shp[1] * shp[2] * shp[3]
        # layer_outs = tf.transpose(layer_outs, [3, 1, 2, 0])
        #
        # v0 = tf.all_variables()
        # layer_outs = self.dense_block(tf.reshape(layer_outs, [shp[3], -1]), 256, activation=tf.nn.tanh)
        # layer_outs = self.dense_block(layer_outs, 32, activation=tf.nn.tanh)
        # layer_outs = self.dense_block(layer_outs, self.num_classes, activation=tf.nn.tanh)
        # layer_outs = tf.expand_dims(tf.reduce_mean(layer_outs, axis=0), 0)

        layer_outs = self.conv_block(layer_outs, self.num_classes, pool_ksize=4, pool_stride=2, block_depth=1,
                                     compute_bn_mean_var=feature_extractor_bn_mean_var_compute)
        layer_outs = tf.reduce_max(tf.reduce_max(layer_outs, axis=1), axis=1)

        v1 = tf.all_variables()
        # restore_excluded_vars = [v for v in v1 if v not in v0]
        restore_excluded_vars = []

        # trainable_vars = [v for v in tf.trainable_variables() if v not in stop_grad_vars_and_ops[0]]

        stop_grad_vars_and_ops = [[], []]
        trainable_vars = tf.trainable_variables()
        return layer_outs, trainable_vars, restore_excluded_vars, stop_grad_vars_and_ops




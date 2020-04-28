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

    def __init__(self, num_classes, im_w=84, im_h=84, compute_bn_mean_var=True, start_step=0, dropout_enabled=False,
                 learn_rate=2.5e-4, l2_regularizer_coeff=1e-2, num_steps=1000000, dropout_rate=.3, discount_factor=.99,
                 update_batchnorm_means_vars=True, optimized_inference=False, load_training_vars=True,
                 model_folder=None, model_prefix='model'):
        if model_folder is None:
            self.model_folder = 'all_trained_models/trained_models'
        else:
            self.model_folder = model_folder
        if not os.path.isdir(self.model_folder):
            os.makedirs(self.model_folder)
        self.model_prefix = model_prefix
        self.model_fpath_prefix = self.model_folder + '/' + model_prefix + '-'
        self.num_classes = num_classes
        self.im_h = im_h
        self.im_w = im_w
        self.compute_bn_mean_var = compute_bn_mean_var
        self.optimized_inference = optimized_inference
        self.x_tensor = tf.placeholder(tf.float32, shape=[None, self.im_h, self.im_w, 4],
                                       name='input_x_tensor')
        self.actions_tensor = tf.placeholder(tf.int32, shape=[None], name='actions_tensor')
        self.layers = [self.x_tensor]

        self.start_step = start_step
        self.step = start_step
        self.learn_rate = learn_rate
        self.step_ph = tf.Variable(self.start_step, trainable=False, name='train_step')
        self.discount_factor_tensor = tf.constant(discount_factor)

        self.learn_rate_tf = tf.Variable(self.learn_rate, trainable=False, name='learn_rate')
        self.unsaved_vars = []

        self.sess = None
        if self.optimized_inference:
            self.dropout_enabled = False
            self.y_pred, _, _, _ = self.init_nn_graph()
            self.outs_softmax_op = tf.nn.softmax(self.y_pred)
            self.outs_final = tf.argmax(self.outs_softmax_op, axis=-1), self.outs_softmax_op, self.y_pred
            self.vars_to_keep = [v for v in tf.global_variables() if v not in self.unsaved_vars]
            self.restorer = tf.train.Saver(var_list=self.vars_to_keep)
            return
        self.dropout_enabled = dropout_enabled
        self.l2_regularizer_coeff = l2_regularizer_coeff
        self.y_gt = tf.placeholder(tf.float32, shape=[None], name='qvals_gt')
        if self.dropout_enabled:
            self.dropout_rate = dropout_rate
            self.dropout_rate_tensor = tf.placeholder(tf.float32, shape=())
        self.y_pred, self.trainable_vars, self.restore_excluded_vars, stop_grad_vars_and_ops = self.init_nn_graph()
        self.outs_softmax_op = tf.nn.softmax(self.y_pred)
        self.actions = tf.argmax(self.y_pred, axis=-1)
        self.outs_final = self.actions, self.outs_softmax_op, self.y_pred

        self.stop_grad_vars, self.stop_grad_update_ops = stop_grad_vars_and_ops

        self.actions_mask_pred = tf.one_hot(self.actions_tensor, self.num_classes)
        y_pred_rewards_mask = self.y_pred * self.actions_mask_pred
        self.q_pred = tf.reduce_max(y_pred_rewards_mask, axis=1) + tf.reduce_min(y_pred_rewards_mask, axis=1)

        self.loss_op = tf.reduce_mean(tf.math.squared_difference(self.q_pred, self.y_gt))

        l2_losses = [self.l2_regularizer_coeff * tf.nn.l2_loss(v) for v in self.trainable_vars]
        self.loss = tf.reduce_mean(self.loss_op)
        self.reduced_loss = self.loss + tf.add_n(l2_losses)
        self.grads_tensor = None
        self.opt = tf.train.RMSPropOptimizer(learning_rate=self.learn_rate_tf, momentum=.95)
        self.train_op = self.opt.minimize(self.reduced_loss, global_step=self.step_ph, var_list=self.trainable_vars,
                                          grad_loss=self.grads_tensor, name='rmsprop_train_op')

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

    def save(self, suffix=None):
        if self.optimized_inference:
            self.restorer.save(self.sess, self.model_prefix)
            print('Model Saved in optimized inference mode')
            return
        if suffix:
            save_fpath = self.model_fpath_prefix + '-' + suffix + '--' + str(self.step)
        else:
            save_fpath = self.model_fpath_prefix + '-' + str(self.step)
        self.saver.save(self.sess, save_fpath)
        print('Model saved at', save_fpath)
        return save_fpath

    def load(self, model_path=None):
        if not self.sess:
            self.init()
        if model_path is None:
            model_folder = self.model_folder
            if 'run-' in self.model_folder:
                curr_run_id = int(self.model_folder.split('run-')[-1].split(os.sep)[0])
                mf = self.model_folder.replace('run-' + str(curr_run_id), 'run-' + str(curr_run_id - 1))
                if os.path.isdir(mf):
                    if len(glob(mf + os.sep + '*')) > 0:
                        model_folder = mf
            if os.path.isdir(model_folder):
                existing_paths = glob(model_folder + '/*.index')
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
        im = im_in / 255.
        # self.viz_inputs(im, 'infer')
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
        im = im_in / 255.
        im = np.expand_dims(im, 0)
        out_label_idx, out_label_conf, out_label_logits = self.sess.run(self.outs_final, feed_dict={self.x_tensor: im})
        return out_label_idx, out_label_conf, out_label_logits

    def viz_inputs(self, x, prefix=''):
        if not os.path.isdir('misc'):
            os.makedirs('misc')
        for b in range(x.shape[0]):
            a = x[b]
            a = (a * 255).astype(np.uint8)
            for i in range(a.shape[-1]):
                cv2.imwrite('misc/' + prefix + str(b) + '-' + str(i) + '.png', a[:, :, i])

    def train_step(self, x_in, y, actions):
        x = x_in / 255.
        # self.viz_inputs(x, 'train')
        if self.dropout_enabled:
            l2_loss, loss, _, step_tf = self.sess.run([self.reduced_loss, self.loss, self.train_op, self.step_ph],
                                                      feed_dict={self.x_tensor: x,
                                                                 self.y_gt: y,
                                                                 self.actions_tensor: actions,
                                                                 self.dropout_rate_tensor: self.dropout_rate})
        else:
            l2_loss, loss, _, step_tf = self.sess.run([self.reduced_loss, self.loss, self.train_op, self.step_ph],
                                                      feed_dict={self.x_tensor: x,
                                                                 self.y_gt: y,
                                                                 self.actions_tensor: actions})
        self.step = step_tf
        return l2_loss, loss, step_tf

    def conv_block(self, x_in, output_filters, kernel_size=3, kernel_stride=1, dilation=1, padding="VALID",
                   batch_norm=False, activation=tf.nn.relu, pooling=False, pool_ksize=3, pool_stride=1,
                   pool_padding="VALID", pooling_fn=tf.nn.max_pool, block_depth=1, make_residual=False,
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

    def dense_block(self, x_in, num_outs, batch_norm=False, activation=tf.nn.relu):
        curr_layer = []
        biased = False
        if not batch_norm:
            biased = True
        layer_outs = tf.layers.dense(x_in, num_outs, use_bias=biased)
        curr_layer.append(layer_outs)
        if activation is not None:
            layer_outs = activation(layer_outs)
        curr_layer.append(layer_outs)
        if batch_norm:
            layer_outs = tf.layers.batch_normalization(layer_outs, training=self.compute_bn_mean_var, axis=-1)
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

    # Architecture from paper https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf
    def init_nn_graph(self):
        feature_extractor_bn_mean_var_compute = False
        layer_outs = self.conv_block(self.x_tensor, 32, kernel_size=8, kernel_stride=4,
                                     compute_bn_mean_var=feature_extractor_bn_mean_var_compute)
        layer_outs = self.conv_block(layer_outs, 64, kernel_size=4, kernel_stride=2,
                                     compute_bn_mean_var=feature_extractor_bn_mean_var_compute)
        layer_outs = self.conv_block(layer_outs, 64, kernel_size=3, kernel_stride=1,
                                     compute_bn_mean_var=feature_extractor_bn_mean_var_compute)

        stop_grad_vars_and_ops = self.__get_nn_vars_and_ops()
        shp = layer_outs.shape
        flat_len = shp[1] * shp[2] * shp[3]
        # layer_outs = tf.transpose(layer_outs, [3, 1, 2, 0])

        v0 = tf.all_variables()
        layer_outs = self.dense_block(tf.reshape(layer_outs, [-1, flat_len]), 512)
        layer_outs = self.dense_block(layer_outs, self.num_classes, activation=None)
        layer_outs = tf.identity(layer_outs, name='qval_pred')

        v1 = tf.all_variables()
        # restore_excluded_vars = [v for v in v1 if v not in v0]
        restore_excluded_vars = []

        # trainable_vars = [v for v in tf.trainable_variables() if v not in stop_grad_vars_and_ops[0]]

        stop_grad_vars_and_ops = [[], []]
        trainable_vars = tf.trainable_variables()
        return layer_outs, trainable_vars, restore_excluded_vars, stop_grad_vars_and_ops




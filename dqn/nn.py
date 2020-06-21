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
import itertools

import tensorflow as tf
import numpy as np
import cv2


class DQN:

    def __init__(self, num_classes, im_w=84, im_h=84, compute_bn_mean_var=True, start_step=0, dropout_enabled=False,
                 learn_rate=2.5e-4, l2_regularizer_coeff=1e-2, num_train_steps=1000000, dropout_rate=.3,
                 discount_factor=.99, update_batchnorm_means_vars=True, optimized_inference=False,
                 load_training_vars=True, model_folder=None, model_prefix='model', other_dqn=None):
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
        self.other_dqn = other_dqn
        self.num_train_steps = num_train_steps
        self.compute_bn_mean_var = compute_bn_mean_var
        self.optimized_inference = optimized_inference

        self.g = tf.Graph()

        with self.g.as_default():
            n0 = len(tf.global_variables())
            self.x_tensor = tf.placeholder(tf.float32, shape=[None, self.im_h, self.im_w, 4])
            self.actions_tensor = tf.placeholder(tf.int32, shape=[None])
            self.layers = [self.x_tensor]
            self.param_tensors_layerwise = [[]]

            self.start_step = start_step
            self.step = start_step
            self.learn_rate = learn_rate
            self.step_ph = tf.Variable(self.start_step, trainable=False)
            self.discount_factor_tensor = tf.constant(discount_factor)

            self.learn_rate_tf = tf.Variable(self.learn_rate, trainable=False)
            self.unsaved_vars = []

            self.sess = None
            if self.optimized_inference:
                self.dropout_enabled = False
                self.y_pred, self.trainable_vars, _ = self.init_nn_graph()
                # if other_dqn is not None:
                #     self.sync_ops = [self.trainable_vars[i].assign(self.other_dqn.trainable_vars[i])
                #                      for i in range(len(self.trainable_vars))]
                self.restorer = tf.train.Saver(var_list=self.trainable_vars)
                self.all_vars = tf.global_variables()[n0:]
                return
            self.dropout_enabled = dropout_enabled
            self.l2_regularizer_coeff = l2_regularizer_coeff
            self.y_gt = tf.placeholder(tf.float32, shape=[None], name='qvals_gt')
            if self.dropout_enabled:
                self.dropout_rate = dropout_rate
                self.dropout_rate_tensor = tf.placeholder(tf.float32, shape=())
            self.y_pred, self.trainable_vars, stop_grad_vars_and_ops = self.init_nn_graph()
            self.outs_softmax_op = tf.nn.softmax(self.y_pred)
            self.actions = tf.argmax(self.y_pred, axis=-1)
            self.outs_final = self.actions, self.outs_softmax_op, self.y_pred

            self.stop_grad_vars, self.stop_grad_update_ops = stop_grad_vars_and_ops

            self.q_pred = tf.reduce_sum(self.y_pred * tf.one_hot(self.actions_tensor, self.num_classes), axis=1)

            self.loss_op = tf.math.squared_difference(self.q_pred, self.y_gt)
            self.learn_rate_tf = tf.train.polynomial_decay(self.learn_rate_tf, global_step=self.step_ph,
                                                           decay_steps=self.num_train_steps, end_learning_rate=1e-6)

            l2_losses = [self.l2_regularizer_coeff * tf.nn.l2_loss(v) for v in self.trainable_vars]
            self.loss = tf.reduce_mean(self.loss_op)
            self.reduced_loss = self.loss + tf.add_n(l2_losses)
            self.opt = tf.train.RMSPropOptimizer(learning_rate=self.learn_rate_tf, decay=.95)
            self.grads_and_vars_tensor = self.opt.compute_gradients(self.loss, var_list=self.trainable_vars)
            self.grads_tensor = [gv[0] for gv in self.grads_and_vars_tensor if gv[0] is not None]
            vars_tensor = [gv[1] for gv in self.grads_and_vars_tensor if gv[0] is not None]
            self.grads_tensor = [tf.clip_by_value(g, -1, 1) for g in self.grads_tensor]
            self.grads_and_vars_tensor = zip(self.grads_tensor, vars_tensor)
            self.train_op = self.opt.apply_gradients(self.grads_and_vars_tensor, global_step=self.step_ph,
                                                     name='rmsprop_train_op')
            self.saver = tf.train.Saver(max_to_keep=0)
            self.restorer = tf.train.Saver()
            self.all_vars = tf.global_variables()[n0:]

    def init(self):
        if not self.sess:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config, graph=self.g)
            init = tf.initialize_variables(self.all_vars)
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
            outs = self.sess.run(self.y_pred, feed_dict={self.x_tensor: im, self.dropout_rate_tensor: 0.})
        else:
            outs = self.sess.run(self.y_pred, feed_dict={self.x_tensor: im})
        return outs

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

    def viz_layer_outs(self, layer_tensor, input_tensor, writeviz=True):
        raw_outs = self.sess.run(layer_tensor, feed_dict={self.x_tensor: input_tensor})
        if len(raw_outs.shape) == 4 and writeviz:
            q = (raw_outs / raw_outs.max()) * 255
            layer_name_parsed = layer_tensor.name.replace('/', '-').replace(':', '_')
            root_dir = 'misc' + os.sep + layer_name_parsed
            num_inputs = q.shape[0]
            num_feature_maps = q.shape[-1]
            if not os.path.isdir(root_dir):
                os.makedirs(root_dir)
            for i in range(num_inputs):
                dirpath = root_dir + os.sep + 'batch_' + str(i)
                if not os.path.isdir(dirpath):
                    os.makedirs(dirpath)
                for j in range(num_feature_maps):
                    fpath = dirpath + os.sep + '-'.join(['feature_' + str(j), 'batch_' + str(i),
                                                         'layer_' + layer_name_parsed]) + '.png'
                    print('writing', fpath)
                    cv2.imwrite(fpath, cv2.resize(q[i, :, :, j], (200, 200), interpolation=cv2.INTER_NEAREST))
        return raw_outs

    def train_step(self, x_in, y, actions):
        x = x_in / 255.
        # self.viz_inputs(x, 'train')
        # self.viz_layer_outs(self.layers[0], x)
        if self.dropout_enabled:
            l2_loss, loss, _, step_tf = self.sess.run([self.reduced_loss, self.loss, self.train_op, self.step_ph],
                                                      feed_dict={self.x_tensor: x,
                                                                 self.y_gt: y,
                                                                 self.actions_tensor: actions,
                                                                 self.dropout_rate_tensor: self.dropout_rate})
        else:
            l2_loss, loss, _, step_tf, \
            grads, q_pred, lr = self.sess.run([self.reduced_loss, self.loss, self.train_op,
                                               self.step_ph, self.grads_tensor, self.q_pred,
                                               self.learn_rate_tf],
                                              feed_dict={self.x_tensor: x,
                                                         self.y_gt: y,
                                                         self.actions_tensor: actions})
        # loss_np = np.mean(np.square(np.array(y) - np.array(q_pred)))
        # gm = np.mean([np.abs(g).mean() for g in grads])
        self.step = step_tf  # Gradients vanishing before train step 11552
        self.learn_rate = lr  # 0 outs at layer 3 <tf.Tensor 'conv2d_5/Relu:0' shape=(?, 7, 7, 64) dtype=float32>
        # self.analyze_layer(x, 3)
        return l2_loss, loss, step_tf

    def analyze_layer(self, x, layer_idx):
        nn_layer_ins, nn_layer_outs_linear, nn_layer_outs_relu \
            = self.sess.run([self.layers[layer_idx - 1][-1], self.layers[layer_idx][0], self.layers[layer_idx][1]],
                            feed_dict={self.x_tensor: x})
        weights = self.param_tensors_layerwise[layer_idx][0].eval(self.sess)
        biases = self.param_tensors_layerwise[layer_idx][1].eval(self.sess)

        np_layer_outs_linear = self.np_convolve(weights, biases, nn_layer_ins, stride=1, activation=False)
        d_linear = np.abs(nn_layer_outs_linear - np_layer_outs_linear)
        e_linear = d_linear.mean()

        np_layer_outs_relu = self.np_convolve(weights, biases, nn_layer_ins, stride=1, activation=True)
        d_relu = np.abs(nn_layer_outs_relu - np_layer_outs_relu)
        e_relu = d_relu.mean()
        print(e_linear, e_relu)

    def np_convolve(self, filters, biases, x_in, stride=1, activation=True):
        n, d_in, _, c_in_x = x_in.shape
        ksize, _, c_in_f, c_out = filters.shape
        in2out_mindim = lambda dim_in: ((dim_in - ksize) // stride) + 1
        out2in_mindim = lambda dim_out: stride * dim_out
        out2in_maxdim = lambda dim_out: out2in_mindim(dim_out) + ksize - 1
        d_out = in2out_mindim(d_in)
        get_receptive_field = lambda x, y: x_in[:, out2in_mindim(y): out2in_maxdim(y) + 1,
                                                out2in_mindim(x): out2in_maxdim(x) + 1, :]

        reshape_rf_for_conv_input = lambda rf: np.rollaxis(np.tile(np.expand_dims(rf, 1), [1, c_out, 1, 1, 1]), 1, 5)
        filters_repeated = np.tile(np.expand_dims(filters, 0), [n, 1, 1, 1, 1])
        biases_repeated = np.tile(np.expand_dims(biases, 0), [n, 1])
        get_conv_output_elemwise = lambda x, y: reshape_rf_for_conv_input(get_receptive_field(x, y)) * filters_repeated
        get_filters_outs = lambda x, y: get_conv_output_elemwise(x, y).reshape([n, -1, c_out]).sum(axis=1)
        get_layer_out_rf = lambda xy: get_filters_outs(xy[0], xy[1]) + biases_repeated
        all_out_xys = list(itertools.product(np.arange(d_out), np.arange(d_out)))
        y = np.array(list(map(get_layer_out_rf, all_out_xys)))
        y = np.transpose(np.rollaxis(y, 0, 2).reshape([n, d_out, d_out, -1]), [0, 2, 1, 3])
        if activation:
            y[y < 0] = 0.
        return y

    def conv_block(self, x_in, output_filters, kernel_size=3, kernel_stride=1, dilation=1, padding="VALID",
                   batch_norm=False, activation=tf.nn.relu, pooling=False, pool_ksize=3, pool_stride=1,
                   pool_padding="VALID", pooling_fn=tf.nn.max_pool, block_depth=1, make_residual=False,
                   compute_bn_mean_var=None):
        num_global_vars_init = len(tf.global_variables())
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
                                         use_bias=use_bias, dilation_rate=dilation,
                                         padding=padding)
            curr_layer.append(layer_out)
            if activation is not None:
                layer_out = activation(layer_out)
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
        self.param_tensors_layerwise.append(tf.global_variables()[num_global_vars_init:])
        return output

    def dense_block(self, x_in, num_outs, batch_norm=False, activation=tf.nn.relu):
        num_global_vars_init = len(tf.global_variables())
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
        self.param_tensors_layerwise.append(tf.global_variables()[num_global_vars_init:])
        return layer_outs

    def __get_nn_vars_and_ops(self):
        vars = tf.global_variables()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        return vars, update_ops

    # Architecture from paper https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf
    def init_nn_graph(self):
        n = len(tf.trainable_variables())
        feature_extractor_bn_mean_var_compute = False
        layer_outs = self.conv_block(self.x_tensor, 32, kernel_size=8, kernel_stride=4,
                                     compute_bn_mean_var=feature_extractor_bn_mean_var_compute)
        layer_outs = self.conv_block(layer_outs, 64, kernel_size=4, kernel_stride=2,
                                     compute_bn_mean_var=feature_extractor_bn_mean_var_compute)
        layer_outs = self.conv_block(layer_outs, 64, kernel_size=3, kernel_stride=1,
                                     compute_bn_mean_var=feature_extractor_bn_mean_var_compute)

        shp = layer_outs.shape
        flat_len = shp[1] * shp[2] * shp[3]

        layer_outs = self.dense_block(tf.reshape(layer_outs, [-1, flat_len]), 512, activation=tf.nn.relu)
        layer_outs = self.dense_block(layer_outs, self.num_classes, activation=None)
        layer_outs = tf.identity(layer_outs, name='qval_pred')
        stop_grad_vars_and_ops = [[], []]
        trainable_vars = tf.trainable_variables()[n:]
        return layer_outs, trainable_vars, stop_grad_vars_and_ops




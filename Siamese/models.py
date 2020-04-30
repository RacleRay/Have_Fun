# -*- encoding:utf-8 -*-

import tensorflow as tf
import numpy as np


class SiameseNN:
    def __init__(self, config):
        self.config = config
        # 输入
        self.add_placeholders()
        # [batch_size, sequence_size, embed_size]
        q_embed, a_embed = self.add_embeddings()
        with tf.variable_scope('siamese') as scope:
            if self.config.network == 'cnn':
                self.q_trans = self.network_cnn(q_embed, reuse=False)
                scope.reuse_variables()
                self.a_trans = self.network_cnn(a_embed, reuse=True)
            elif self.config.network == 'rnn':
                self.q_trans = self.network_rnn(q_embed)
                tf.get_variable_scope().reuse_variables()
                self.a_trans = self.network_rnn(a_embed)
        # 损失和精确度
        self.total_loss = self.add_loss_op(self.q_trans, self.a_trans)
        # 训练节点
        self.train_op = self.add_train_op(self.total_loss)

    def add_placeholders(self):
        # 问题
        self.q = tf.placeholder(tf.int32,
                shape=[None, self.config.max_q_length],
                name='Question')
        # 回答
        self.a = tf.placeholder(tf.int32,
                shape=[None, self.config.max_a_length],
                name='Answer')
        self.y = tf.placeholder(tf.float32, shape=[None, ], name='label')
        # drop_out
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.batch_size = tf.shape(self.q)[0]

    def add_embeddings(self):
        with tf.variable_scope('embedding'):
            if self.config.embeddings is not None:
                embeddings = tf.Variable(self.config.embeddings, name="embeddings", trainable=False)
            else:
                embeddings = tf.get_variable('embeddings',
                    shape=[self.config.vocab_size, self.config.embedding_size],
                    initializer=tf.uniform_unit_scaling_initializer())
            q_embed = tf.nn.embedding_lookup(embeddings, self.q)
            a_embed = tf.nn.embedding_lookup(embeddings, self.a)
            q_embed = tf.nn.dropout(q_embed, keep_prob=self.keep_prob)
            a_embed = tf.nn.dropout(a_embed, keep_prob=self.keep_prob)
            return q_embed, a_embed

    def network_cnn(self, x, reuse=False):
        # (batch_size, conv_size)
        conv1 = self.conv_layer(x, reuse=reuse)
        # (batch_size, hidden_size)
        fc1 = self.fc_layer(conv1, self.config.hidden_size, "fc1")
        ac1 = tf.nn.relu(fc1)
        # (batch_size, output_size)
        fc2 = self.fc_layer(ac1, self.config.output_size, "fc2")
        return fc2

    def network_rnn(self, x):
        sequence_length = x.get_shape()[1]
        # (batch_size, time_step, embed_size) -> (time_step, batch_size, embed_size)
        inputs = tf.transpose(x, [1, 0, 2])
        # (batch_size, rnn_output_size)
        rnn1 = self.rnn_layer(inputs)
        # (batch_size, hidden_size)
        fc1 = self.fc_layer(rnn1, self.config.hidden_size, "fc1")
        ac1 = tf.nn.relu(fc1)
        # (batch_size, output_size)
        fc2 = self.fc_layer(ac1, self.config.output_size, "fc2")
        return fc2

    def fc_layer(self, inputs, n_dim_output, name):
        assert len(inputs.get_shape()) == 2
        n_prev_weight = inputs.get_shape()[1]
        initializer = tf.truncated_normal_initializer(stddev=0.01)
        W = tf.get_variable(name+'W',
                            dtype=tf.float32,
                            shape=[n_prev_weight, n_dim_output],
                            initializer=initializer)
        b = tf.get_variable(name+'b',
                            dtype=tf.float32,
                            initializer=tf.constant(0.01, shape=[n_dim_output],
                            dtype=tf.float32))
        fc = tf.nn.bias_add(tf.matmul(inputs, W), b)
        return fc

    def conv_layer(self, inputs, reuse=False):
        pool = []
        max_len = inputs.get_shape()[1]
        for i, filter_size in enumerate(self.config.filter_sizes):
            with tf.variable_scope('filter{filter_size}'):
                conv1_W = tf.get_variable('conv_W',
                                          shape=[filter_size, self.config.embedding_size, 1, self.config.num_filters],
                                          initializer=tf.initializers.glorot_uniform())
                conv1_b = tf.get_variable('conv_b',
                                          initializer=tf.constant(0.0, shape=[self.config.num_filters]))
                pool_b = tf.get_variable('pool_b',
                                         initializer=tf.constant(0.0, shape=[2 * self.config.num_filters]))
                # 卷积
                out = tf.nn.relu((tf.nn.conv2d(inputs, conv1_W, [1,1,1,1], padding='VALID') + conv1_b))
                # 池化
                out1 = tf.nn.max_pool(out, [1,max_len-filter_size+1,1,1], [1,1,1,1], padding='VALID')
                out2 = tf.nn.avg_pool(out, [1,max_len-filter_size+1,1,1], [1,1,1,1], padding='VALID')
                out = tf.concat(3, [out1, out1])
                out = tf.nn.tanh(out + pool_b)
                pool.append(out)
                # 加入正则项
                if not reuse:
                    tf.add_to_collection('total_loss', 0.5 * self.config.l2_reg_lambda * tf.nn.l2_loss(conv1_W))

        total_channels = len(self.config.filter_sizes) * self.config.num_filters * 2
        real_pool = tf.reshape(tf.concat(pool, 2), [self.batch_size, total_channels])
        return real_pool

    def rnn_layer(self, inputs):
        if self.config.cell_type == 'lstm':
            birnn_fw, birnn_bw = self.bi_lstm(self.config.rnn_size,
                                            self.config.layer_size,
                                            self.config.keep_prob)
        else:
            birnn_fw, birnn_bw = self.bi_gru(self.config.rnn_size,
                                            self.config.layer_size,
                                            self.config.keep_prob)
        outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(birnn_fw, birnn_bw, inputs, dtype=tf.float32)
        # (time_step, batch_size, 2*rnn_size) -> (batch_size, 2*rnn_size)
        output = tf.reduce_mean(outputs, 0)
        return output

    def bi_lstm(self, rnn_size, layer_size, keep_prob):
        # forward rnn
        with tf.name_scope('fw_rnn'), tf.variable_scope('fw_rnn'):
            lstm_fw_cell_list = [tf.contrib.rnn.LSTMCell(rnn_size) for _ in range(layer_size)]
            lstm_fw_cell_m = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.MultiRNNCell(lstm_fw_cell_list),
                                                            output_keep_prob=keep_prob)
        # backward rnn
        with tf.name_scope('bw_rnn'), tf.variable_scope('bw_rnn'):
            lstm_bw_cell_list = [tf.contrib.rnn.LSTMCell(rnn_size) for _ in range(layer_size)]
            lstm_bw_cell_m = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.MultiRNNCell(lstm_fw_cell_list),
                                                            output_keep_prob=keep_prob)
        return lstm_fw_cell_m, lstm_bw_cell_m

    def bi_gru(self, rnn_size, layer_size, keep_prob):
        # forward rnn
        with tf.name_scope('fw_rnn'), tf.variable_scope('fw_rnn'):
            gru_fw_cell_list = [tf.contrib.rnn.GRUCell(rnn_size) for _ in range(layer_size)]
            gru_fw_cell_m = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.MultiRNNCell(gru_fw_cell_list),
                                                            output_keep_prob=keep_prob)
        # backward rnn
        with tf.name_scope('bw_rnn'), tf.variable_scope('bw_rnn'):
            gru_bw_cell_list = [tf.contrib.rnn.GRUCell(rnn_size) for _ in range(layer_size)]
            gru_bw_cell_m = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.MultiRNNCell(gru_bw_cell_list),
                                                            output_keep_prob=keep_prob)
        return gru_fw_cell_m, gru_bw_cell_m

    def add_loss_op(self, represent1, represent2):
        n_represent1 = tf.nn.l2_normalize(represent1, dim=1)
        n_represent2 = tf.nn.l2_normalize(represent2, dim=1)
        self.q_a_cosine = tf.reduce_sum(tf.multiply(n_represent1, n_represent2), 1)

        loss = self.contrastive_loss(self.q_a_cosine, self.y)
        tf.add_to_collection('total_loss', loss)
        total_loss = tf.add_n(tf.get_collection('total_loss'))
        return total_loss

    def contrastive_loss(self, y_hat, y):
        l_1 = self.config.pos_weight * tf.square(1 - y_hat)
        l_0 = tf.square(tf.maximum(y_hat, 0))
        loss = tf.reduce_mean((1 - y) * l_0 + y * l_1)
        return loss

    def add_train_op(self, loss):
        with tf.name_scope('train_op'):
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            optimizer = tf.train.AdamOptimizer(self.config.lr)
            gradients, variables = zip(*optimizer.compute_gradients(loss))
            gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
            train_op = optimizer.apply_gradients(zip(gradients, variables))
        return train_op
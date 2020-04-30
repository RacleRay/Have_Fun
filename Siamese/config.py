# -*- encoding:utf-8 -*-

import tensorflow as tf


class CNNConfig:
    def __init__(self, vocab_size, embeddings=None):
        self.network = 'cnn'
        self.max_q_length = 25
        self.max_a_length = 90
        # 输入问题(句子)长度
        self.max_q_length = 200
        # 输入答案长度
        self.max_a_length = 200
        # 循环数
        self.num_epochs = 100
        # batch大小
        self.batch_size = 128
        # 词表大小
        self.vocab_size = vocab_size
        # 词向量大小
        self.embeddings = embeddings
        self.embedding_size = 100
        if self.embeddings is not None:
            self.embedding_size = embeddings.shape[1]
        # 不同类型的filter，对应不同的尺寸
        self.filter_sizes = [1, 2, 3, 5, 7, 9]
        # 隐层大小
        self.hidden_size = 128
        self.output_size = 128
        # 每种filter的数量
        self.num_filters = 128
        self.l2_reg_lambda = 0.
        self.keep_prob = 0.6
        # 学习率
        self.lr = 0.0003
        # contrasive loss 中的 positive loss部分的权重
        self.pos_weight = 5

        self.cf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        self.cf.gpu_options.per_process_gpu_memory_fraction = 0.2


class RNNConfig:
    def __init__(self, vocab_size, embeddings=None):
        self.network = 'rnn'
        self.max_q_length = 25
        self.max_a_length = 90
        # 输入问题(句子)长度
        self.max_q_length = 200
        # 输入答案长度
        self.max_a_length = 200
        # 循环数
        self.num_epochs = 100
        # batch大小
        self.batch_size = 128
        # 词表大小
        self.vocab_size = vocab_size
        # 词向量大小
        self.embeddings = embeddings
        self.embedding_size = 100
        if self.embeddings is not None:
            self.embedding_size = embeddings.shape[1]
        # RNN单元类型和大小与堆叠层数
        self.cell_type = 'GRU'
        self.rnn_size = 128
        self.layer_size = 1
        # 隐层大小
        self.hidden_size = 128
        self.output_size = 64

        self.keep_prob = 0.8
        # 学习率
        self.lr = 0.0003
        # contrasive loss 中的 positive loss部分的权重
        self.pos_weight = 5

        self.cf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        self.cf.gpu_options.per_process_gpu_memory_fraction = 0.2
# -*- coding:utf-8 -*-
# author: Racle
# project: LdaVec

import tensorflow as tf
import numpy as np

from config import *


class LDA2VEC:
    def __init__(self,
                 num_unique_documents,
                 vocab_size,
                 num_topics,
                 freqs,
                 embedding_size=128,
                 num_sampled=40,
                 learning_rate=1e-3,
                 lmbda=150.0,
                 alpha=None,
                 power=0.75,
                 batch_size=32,
                 clip_gradients=5.0,
                 **kwargs):
        moving_avgs = tf.train.ExponentialMovingAverage(0.9)
        self.batch_size = batch_size
        self.freqs = freqs  # 狄利克雷先验
        self.sess = tf.InteractiveSession()

        self.X = tf.placeholder(tf.int32, shape=[None])
        self.Y = tf.placeholder(tf.int64, shape=[None])
        self.DOC = tf.placeholder(tf.int32, shape=[None])  # batch的文档idx
        step = tf.Variable(0, trainable=False, name='global_step')
        self.switch_loss = tf.Variable(0, trainable=False)

        train_labels = tf.reshape(self.Y, [-1, 1])

        # 使用固定的基本分布对一组类进行采样。
        # 从整数范围[0,range_max]中随机采样num_sampled个类，所有的类的类别是[0, range_max),
        # 每个类被采样的概率大小由参数unigrams指定，这个参数的值可以是概率的array，
        # 也可以是int的vector(表示出现次数,次数大表示被采样的概率大)
        # 通过distortion power(失真功率)来调整采样分布.
        # num_true：int,每个训练示例的目标类数
        sampler = tf.nn.fixed_unigram_candidate_sampler(
            train_labels,
            num_true=1,
            num_sampled=num_sampled,
            unique=True,
            range_max=vocab_size,
            distortion=power,
            unigrams=self.freqs,
        )

        self.word_embedding = tf.Variable(
            tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
        self.nce_weights = tf.Variable(
            tf.truncated_normal(
                [vocab_size, embedding_size],
                stddev=tf.sqrt(1 / embedding_size),
            ))
        self.nce_biases = tf.Variable(tf.zeros([vocab_size]))
        scalar = 1 / np.sqrt(num_unique_documents + num_topics)

        # 随机初始化的文档主题分布
        self.doc_embedding = tf.Variable(
            tf.random_normal(
                [num_unique_documents, num_topics],
                mean=0,
                stddev=50 * scalar,
            ))
        # 每个主题的向量表示
        self.topic_embedding = tf.get_variable(
            'topic_embedding',
            shape=[num_topics, embedding_size],
            dtype=tf.float32,
            initializer=tf.orthogonal_initializer(gain=scalar),
        )

        # skipgram中心词
        pivot = tf.nn.embedding_lookup(self.word_embedding, self.X)
        # 文档的topic分布向量
        proportions = tf.nn.embedding_lookup(self.doc_embedding, self.DOC)
        # 文档的topic embedding表达
        doc = tf.matmul(proportions, self.topic_embedding)

        # 结合词向量和文档主题信息的表达方式
        doc_context = doc
        word_context = pivot
        context = tf.add(word_context, doc_context)

        # 负采样计算损失函数
        # sampler为目标类(true_classes)和采样类(sampled_candidates)出现的次数
        loss_word2vec = tf.reduce_mean(
            tf.nn.nce_loss(
                weights=self.nce_weights,
                biases=self.nce_biases,
                labels=self.Y,
                inputs=context,
                num_sampled=num_sampled,
                num_classes=vocab_size,
                num_true=1,
                sampled_values=sampler,
            ))
        self.fraction = tf.Variable(1, trainable=False, dtype=tf.float32)

        # 计算topic分布的损失
        n_topics = self.doc_embedding.get_shape()[1].value
        log_proportions = tf.nn.log_softmax(self.doc_embedding)
        if alpha is None:
            alpha = 1.0 / n_topics
        loss = -(alpha - 1) * log_proportions
        prior = tf.reduce_sum(loss)

        loss_lda = lmbda * self.fraction * prior  # topic分布的损失
        self.cost = tf.cond(
            step < self.switch_loss,
            lambda: loss_word2vec,
            lambda: loss_word2vec + loss_lda,
        )

        # Exponential Moving Average
        loss_avgs_op = moving_avgs.apply([loss_lda, loss_word2vec, self.cost])
        with tf.control_dependencies([loss_avgs_op]):
            self.optimizer = tf.contrib.layers.optimize_loss(
                self.cost,
                tf.train.get_global_step(),
                learning_rate,
                'Adam',
                clip_gradients=clip_gradients,
            )
        self.sess.run(tf.global_variables_initializer())

    def train(self,
              pivot_words,
              target_words,
              doc_ids,
              num_epochs,
              switch_loss=3):
        from tqdm import tqdm

        temp_fraction = self.batch_size / len(pivot_words)
        self.sess.run(tf.assign(self.fraction, temp_fraction))
        self.sess.run(tf.assign(self.switch_loss, switch_loss))
        for e in range(num_epochs):
            pbar = tqdm(
                range(0, len(pivot_words), self.batch_size),
                desc='minibatch loop',
            )
            for i in pbar:
                batch_x = pivot_words[i:min(i +
                                            self.batch_size, len(pivot_words))]
                batch_y = target_words[i:min(i +
                                             self.batch_size, len(pivot_words
                                                                  ))]
                batch_doc = doc_ids[i:min(i +
                                          self.batch_size, len(pivot_words))]
                _, cost = self.sess.run(
                    [self.optimizer, self.cost],
                    feed_dict={
                        self.X: batch_x,
                        self.Y: batch_y,
                        self.DOC: batch_doc,
                    },
                )
                pbar.set_postfix(cost=cost, epoch=e + 1)

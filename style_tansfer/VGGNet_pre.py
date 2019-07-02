#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import math
import numpy as np
import tensorflow as tf


class VGGNet:
    """
    Builds VGG-16 net structure,
    load parameters from pre-train models.
    https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM
    模型文件是预训练好的文件，但是文件是npy文件，不是tensorflow的模型文件。
    """
    def __init__(self, path):
        self._vgg16_data = np.load(path, encoding='latin1')
        self.data_dict = self._vgg16_data.item()
        # 归一化时RGB通道上的均值
        sefl._VGG_MEAN = [103.939, 116.779, 123.68]
    
    def get_conv_filter(self, name):
        '''name: 'conv1_1', 'conv1_2'...'''
        # tf.constant相当于trainable=False一样
        return tf.constant(self.data_dict[name][0], name='conv')

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name='fc')
    
    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name='bias')

    def conv_layer(self, x, name):
        """Builds convolution layer."""
        with tf.name_scope(name):
            conv_w = self.get_conv_filter(name)
            conv_b = self.get_bias(name)
            # 相比于layers更底层的卷积方式，可以传入自定义参数
            h = tf.nn.conv2d(x, conv_w, [1,1,1,1], padding='SAME')
            h = tf.nn.bias_add(h, conv_b)
            h = tf.nn.relu(h)
            return h

    def pooling_layer(self, x, name):
        """Builds pooling layer."""
        return tf.nn.max_pool(x,
                              ksize = [1,2,2,1],
                              strides = [1,2,2,1],
                              padding = 'SAME',
                              name = name)

    def fc_layer(self, x, name, activation=tf.nn.relu):
        """Builds fully-connected layer."""
        with tf.name_scope(name):
            fc_w = self.get_fc_weight(name)
            fc_b = self.get_bias(name)
            h = tf.matmul(x, fc_w)
            h = tf.nn.bias_add(h, fc_b)
            # 输出值进入softmax，不需要进行activation
            if activation is None:
                return h
            else:
                return activation(h)

    def flatten_layer(self, x, name):
        """Builds flatten layer."""
        with tf.name_scope(name):
            # [batch_size, image_width, image_height, channel]
            x_shape = x.get_shape().as_list()
            dim = 1
            for d in x_shape[1:]:
                dim *= d
            x = tf.reshape(x, [-1, dim])
            return x

    def build(self, x_rgb):
        """Build VGG16 network structure.
        Parameters:
        - x_rgb: [1, 224, 224, 3]
        """
        start_time = time.time()
        print('building model ...')
        
        # 首先处理输入数据，减去VGG_MEAN(切分三个通道)
        r, g, b = tf.split(x_rgb, [1,1,1], axis=3)
        # VGG按照BGR排列
        x_bgr = tf.concat(
                        [b - self.VGG_MEAN[0],
                         g - self.VGG_MEAN[1],
                         r - self.VGG_MEAN[2]],
                        axis=3)

        assert x_bgr.get_shape().as_list()[1:] == [224, 224, 3]

        # conv1_1这里的名字需要和npy文件的keys对应
        # 每一层输出定义到成员变量中，在进行风格转换计算时可以直接获得
        self.conv1_1 = self.conv_layer(x_bgr, 'conv1_1')
        self.conv1_2 = self.conv_layer(self.conv1_1, 'conv1_2')
        self.pool1 = self.pooling_layer(self.conv1_2, 'pool1')
        
        self.conv2_1 = self.conv_layer(self.pool1, 'conv2_1')
        self.conv2_2 = self.conv_layer(self.conv2_1, 'conv2_2')
        self.pool2 = self.pooling_layer(self.conv2_2, 'pool2')
        
        self.conv3_1 = self.conv_layer(self.pool2, 'conv3_1')
        self.conv3_2 = self.conv_layer(self.conv3_1, 'conv3_2')
        self.conv3_3 = self.conv_layer(self.conv3_2, 'conv3_3')
        self.pool3 = self.pooling_layer(self.conv3_3, 'pool3')
        
        self.conv4_1 = self.conv_layer(self.pool3, 'conv4_1')
        self.conv4_2 = self.conv_layer(self.conv4_1, 'conv4_2')
        self.conv4_3 = self.conv_layer(self.conv4_2, 'conv4_3')
        self.pool4 = self.pooling_layer(self.conv4_3, 'pool4')
        
        self.conv5_1 = self.conv_layer(self.pool4, 'conv5_1')
        self.conv5_2 = self.conv_layer(self.conv5_1, 'conv5_2')
        self.conv5_3 = self.conv_layer(self.conv5_2, 'conv5_3')
        self.pool5 = self.pooling_layer(self.conv5_3, 'pool5')

        # # 在风格转换算法中，并不需要全连接层的数据，并且在构建计算图时在这部分
        # # 的文件读取和变量赋值操作很耗时（参数量很大），所以不考虑
        # # 21s到0s（不到1s）的缩减时间
        # self.flatten5 = self.flatten_layer(self.pool5, 'flatten')
        # self.fc6 = self.fc_layer(self.flatten5, 'fc6')
        # self.fc7 = self.fc_layer(self.fc6, 'fc7')
        # # 输出值进入softmax，不需要进行activation
        # self.fc8 = self.fc_layer(self.fc7, 'fc8', activation=None)
        # self.prob = tf.nn.softmax(self.fc8, name='prob')

        print('building model finished: %4ds' % (time.time() - start_time))
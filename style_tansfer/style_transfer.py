#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from VGGNet_pre import VGGNet


class style_transfer_v1:
    
    def __init__(self, content_img_path, style_img_path, 
                 vgg_model_path,
                 num_steps=100, 
                 learning_rate=10, 
                 lambda_c=0.1, 
                 lambda_s=500, 
                 output_dir):
    """lambda_c：loss content系数
       lambda_s：loss style系数
       loss style在计算gram矩阵的时候除以了维度constant，导致数量级较小"""
        self._content_img_path = content_img_path
        self._style_img_path = style_img_path
        self._vgg_model_path = vgg_model_path
        self._num_steps = num_steps
        self._learning_rate = learning_rate
        self._lambda_c = lambda_c
        self._lambda_s = lambda_s
        sefl._output_dir = output_dir

    @staticmethod
    def initial_result(shape, mean, stddev):
        """随机初始化一张图片，通过学习过程一步步更新"""
        initial = tf.truncated_normal(shape, mean=mean, stddev=stddev)
        return tf.Variable(initial)

    @staticmethod
    def read_img(img_name):
        img = Image.open(img_name)
        np_img = np.array(img)  # (224, 224, 3)
        np_img = np.asarray([np_img], dtype=np.int32) # (1, 224, 224, 3)
                                                      # 将[]列表这一维加入
        return np_img

    @staticmethod
    def gram_matrix(x):
        """Calulates gram matrix
        Args:
        - x: feaures extracted from VGG Net. shape: [1, width, height, ch]
        """
        b, w, h, ch = x.get_shape().as_list()
        features =  tf.reshape(x, [b, w*h, ch])   # [ch, ch] -> (i, j)
        # 选择两列计算余弦相似度
        # [h*w, ch] matrix -> [ch, h*w] * [h*w, ch] -> [ch, ch]
        # 除以维度，防止(高维造成的)过大
        gram = tf.matmul(features, features, adjoint_a=True) \
               / tf.constant(ch * w * h, tf.float32) 
        return gram

    def style_transfer_graph(self):
        # 重置默认的图
        tf.reset_default_graph()
        # 定义图的基本信息
        with tf.Graph().as_default() as graph_default:
            # 初始化目标图像
            result = self.initial_result((1, 224, 224, 3), 127.5, 20)

            # 创建placeholder，通过feed_dict传入
            content = tf.placeholder(tf.float32, shape=[1, 224, 224, 3])
            style = tf.placeholder(tf.float32, shape=[1, 224, 224, 3])

            # 通过三个VGG net提取三个图片特征
            vgg_for_content = VGGNet(vgg_model_path)
            vgg_for_style = VGGNet(vgg_model_path)
            vgg_for_result = VGGNet(vgg_model_path)

            # 输入图片数据
            vgg_for_content.build(content)
            vgg_for_style.build(style)
            vgg_for_result.build(result)

            # content特征部分
            # content_features在离输入层近的层会比较接近content原图（抽象程度更高）
            # 可以使用多层
            content_features = [
                vgg_for_content.conv1_2,
                vgg_for_content.conv2_2,
                # vgg_for_content.conv3_3,
                # vgg_for_content.conv4_3,
                # vgg_for_content.conv5_3
                ]

            result_content_features = [
                vgg_for_result.conv1_2,
                vgg_for_result.conv2_2,
                # vgg_for_result.conv3_3,
                # vgg_for_result.conv4_3,
                # vgg_for_result.conv5_3
                ]

            # style特征部分(第五层学习效果较差)
            # feature_size: [1, width, height, channel]
            # 可以使用多层
            style_features = [
                # vgg_for_style.conv1_2,
                # vgg_for_style.conv2_2,
                # vgg_for_style.conv3_3,
                vgg_for_style.conv4_3,
                # vgg_for_style.conv5_3
                ]
            # 每个channel两两计算余弦相似度构成gram矩阵
            style_gram = [self.gram_matrix(feature) for feature in style_features]

            # result_style_features需要和style_features的层次保持一致
            result_style_features = [
                # vgg_for_result.conv1_2,
                # vgg_for_result.conv2_2,
                # vgg_for_result.conv3_3,
                vgg_for_result.conv4_3,
                # vgg_for_result.conv5_3
                ]
            result_style_gram = [self.gram_matrix(feature) for feature in result_style_features]

            # 损失函数计算
            # content_loss可以对多层求，并添加各层权重（添加一个列表）
            content_loss = tf.zeros(1, tf.float32)
            # shape: [1, width, height, channel]
            # 对应层次对应位置配对成元组
            for c, c_ in zip(content_features, result_content_features):
                content_loss += tf.reduce_mean((c - c_) ** 2, [1, 2, 3])

            # style_loss
            # style_loss可以对多层求，并添加各层权重（添加一个列表）
            style_loss = tf.zeros(1, tf.float32)
            for s, s_ in zip(style_gram, result_style_gram):
                style_loss += tf.reduce_mean((s - s_) ** 2, [1, 2])

            loss = content_loss * self._lambda_c + style_loss * self._lambda_s
            with tf.name_scope('train_op'):
                train_op = tf.train.AdamOptimizer(self._learning_rate).minimize(loss)

            init_op = tf.global_variables_initializer()

    def train(self):
        # 读取内容图像，内容图像
        content_val = self.read_img(self._content_img_path)
        style_val = self.read_img(self._style_img_path)

        with tf.Session(graph=graph_default) as sess:
            sess.run(init_op)
            for step in range(self._num_steps):
                loss_value, content_loss_value, style_loss_value, _ = \
                    sess.run([loss, content_loss, style_loss, train_op],
                            feed_dict={
                                content: content_val,
                                style: style_val,
                            })
                # loss_value,content_loss_value,style_loss_value创建为
                # tf.zeros(1, tf.float32)数组，[0]取出value
                print(
                    'step: %d, loss_value: %8.4f, content_loss: %8.4f, style_loss: %8.4f'
                    % (step + 1, 
                       loss_value[0], 
                       content_loss_value[0],
                       style_loss_value[0])
                    )
                # 输出生成的图像
                result_img_path = os.path.join(
                    self._output_dir, 'result-%05d.jpg' % (step + 1))
                result_val = result.eval(sess)[0]
                result_val = np.clip(result_val, 0, 255)
                img_arr = np.asarray(result_val, np.uint8)
                img = Image.fromarray(img_arr)
                img.save(result_img_path)


if __name__ == '__main__':
    vgg_model_path = './style_transfer_data/vgg16.npy'
    content_img_path = './style_transfer_data/gugong.jpg'
    style_img_path = './style_transfer_data/xingkong.jpeg'
    output_dir = './run_style_transfer'

    model = style_transfer_v1(content_img_path, style_img_path, 
                 vgg_model_path, output_dir=output_dir)
    model.style_transfer_graph()
    model.train()

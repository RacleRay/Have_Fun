import tensorflow as tf
import numpy as np
from scipy.misc import imresize
from imageio import imread, imsave, mimsave
import glob
import os
from helper import montage

batch_size = 100
z_dim = 100
WIDTH = 64
HEIGHT = 64

dataset = 'lfw_new_imgs' # LFW
# dataset = 'celeba' # CelebA
images = glob.glob(os.path.join(dataset, '*.*'))

OUTPUT_DIR = 'samples_' + dataset
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

X = tf.placeholder(dtype=tf.float32, shape=[None, HEIGHT, WIDTH, 3], name='X')
noise = tf.placeholder(dtype=tf.float32, shape=[None, z_dim], name='noise')
is_training = tf.placeholder(dtype=tf.bool, name='is_training')


def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)


def discriminator(image, reuse=None, is_training=is_training):
    momentum = 0.9
    with tf.variable_scope('discriminator', reuse=reuse):
        h0 = lrelu(tf.layers.conv2d(image, kernel_size=5, filters=64, strides=2, padding='same'))

        h1 = tf.layers.conv2d(h0, kernel_size=5, filters=128, strides=2, padding='same')
        h1 = lrelu(tf.contrib.layers.batch_norm(h1, is_training=is_training, decay=momentum))

        h2 = tf.layers.conv2d(h1, kernel_size=5, filters=256, strides=2, padding='same')
        h2 = lrelu(tf.contrib.layers.batch_norm(h2, is_training=is_training, decay=momentum))

        h3 = tf.layers.conv2d(h2, kernel_size=5, filters=512, strides=2, padding='same')
        h3 = lrelu(tf.contrib.layers.batch_norm(h3, is_training=is_training, decay=momentum))

        h4 = tf.contrib.layers.flatten(h3)
        h4 = tf.layers.dense(h4, units=1)
        return tf.nn.sigmoid(h4), h4


def generator(z, is_training=is_training):
    momentum = 0.9
    with tf.variable_scope('generator', reuse=None):
        d = 4
        h0 = tf.layers.dense(z, units=d * d * 512)
        h0 = tf.reshape(h0, shape=[-1, d, d, 512])
        h0 = tf.nn.relu(tf.contrib.layers.batch_norm(h0, is_training=is_training, decay=momentum))

        h1 = tf.layers.conv2d_transpose(h0, kernel_size=5, filters=256, strides=2, padding='same')
        h1 = tf.nn.relu(tf.contrib.layers.batch_norm(h1, is_training=is_training, decay=momentum))

        h2 = tf.layers.conv2d_transpose(h1, kernel_size=5, filters=128, strides=2, padding='same')
        h2 = tf.nn.relu(tf.contrib.layers.batch_norm(h2, is_training=is_training, decay=momentum))

        h3 = tf.layers.conv2d_transpose(h2, kernel_size=5, filters=64, strides=2, padding='same')
        h3 = tf.nn.relu(tf.contrib.layers.batch_norm(h3, is_training=is_training, decay=momentum))

        h4 = tf.layers.conv2d_transpose(h3, kernel_size=5, filters=3, strides=2, padding='same', activation=tf.nn.tanh, name='g')
        return h4


# 判别两种图
g = generator(noise)
d_real, d_real_logits = discriminator(X)
d_fake, d_fake_logits = discriminator(g, reuse=True)

# 分别记录变量
vars_g = [var for var in tf.trainable_variables() if var.name.startswith('generator')]
vars_d = [var for var in tf.trainable_variables() if var.name.startswith('discriminator')]

# 损失函数
# 判别器尽量使d_real_logits与1的距离小，使d_fake_logits与0的距离越小：即准确辨别真假
loss_d_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(d_real_logits, tf.ones_like(d_real)))
loss_d_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(d_fake_logits, tf.zeros_like(d_fake)))
loss_d = loss_d_real + loss_d_fake
# 生成器尽量使d_fake_logits与1距离小
loss_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(d_fake_logits, tf.ones_like(d_fake)))


# 分离判别器与生成器的训练
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer_d = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(loss_d, var_list=vars_d)
    optimizer_g = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(loss_g, var_list=vars_g)










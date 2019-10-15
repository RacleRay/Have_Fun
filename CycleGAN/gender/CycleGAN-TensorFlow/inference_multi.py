# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from model import CycleGAN
from imageio import imread, imsave
import glob
import os

# 中间输出文件
image_file = 'face.jpg'
W = 256
# 4行5列
result = np.zeros((4 * W, 5 * W, 3))

for gender in ['male', 'female']:
    if gender == 'male':
        images = glob.glob('../data/male/*.jpg')
        model = '../pretrained/male2female.pb'
        r = 0  # 前两行：一行male原始，一行转换后male
    else:
        images = glob.glob('../data/female/*.jpg')
        model = '../pretrained/female2male.pb'
        r = 2  # 后两行：一行female原始，一行转换后female

    graph = tf.Graph()
    with graph.as_default():
        graph_def = tf.GraphDef()
        with tf.gfile.FastGFile(model, 'rb') as model_file:
            graph_def.ParseFromString(model_file.read())
            tf.import_graph_def(graph_def, name='')

        with tf.Session(graph=graph) as sess:
            input_tensor = graph.get_tensor_by_name('input_image:0')
            output_tensor = graph.get_tensor_by_name('output_image:0')

            for i, image in enumerate(images):
                image = imread(image)
                # 此时output是一个字符串
                output = sess.run(output_tensor, feed_dict={input_tensor: image})

                with open(image_file, 'wb') as f:
                    f.write(output)

                # 此时output是一个nparray
                output = imread(image_file)
                maxv = np.max(output)
                minv = np.min(output)
                output = ((output - minv) / (maxv - minv) * 255).astype(np.uint8)

                # 前两行：一行male原始，一行转换后male
                # 后两行：一行female原始，一行转换后female
                result[r * W: (r + 1) * W, i * W: (i + 1) * W, :] = image
                result[(r + 1) * W: (r + 2) * W, i * W: (i + 1) * W, :] = output

os.remove(image_file)
imsave('CycleGAN_result.jpg', result)
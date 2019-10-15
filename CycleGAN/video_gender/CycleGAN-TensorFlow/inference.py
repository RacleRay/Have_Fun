"""Translate an image to another image
An example of command-line usage is:
python export_graph.py --model pretrained/apple2orange.pb \
                       --input input_sample.jpg \
                       --output output_sample.jpg \
                       --image_size 256
"""

import tensorflow as tf
import os
from model import CycleGAN
import utils
import numpy as np
from imageio import imread, imsave
import cv2

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('model', '', 'model path (.pb)')
tf.flags.DEFINE_string('input', 'input_sample.jpg', 'input image path (.jpg)')
tf.flags.DEFINE_string('output', 'output_sample.jpg', 'output image path (.jpg)')
tf.flags.DEFINE_integer('image_size', '256', 'image size, default: 256')

def inference():
  input_image = imread(FLAGS.input)
  h = input_image.shape[0]
  w = input_image.shape[1]
  if h > w:
    input_image = input_image[h // 2 - w // 2: h // 2 + w // 2, :, :]
  else:
    input_image = input_image[:, w // 2 - h // 2: w // 2 + h // 2, :]    
  input_image = cv2.resize(input_image, (FLAGS.image_size, FLAGS.image_size))

  graph = tf.Graph()
  with graph.as_default():
    graph_def = tf.GraphDef()
    with tf.gfile.FastGFile(FLAGS.model, 'rb') as model_file:
      graph_def.ParseFromString(model_file.read())
      tf.import_graph_def(graph_def, name='')

    with tf.Session(graph=graph) as sess:
      input_tensor = graph.get_tensor_by_name('input_image:0')
      output_tensor = graph.get_tensor_by_name('output_image:0')
      output_image = sess.run(output_tensor, feed_dict={input_tensor: input_image})
      with open(FLAGS.output, 'wb') as f:
        f.write(output_image)

      output_image = imread(FLAGS.output)
      maxv = np.max(output_image)
      minv = np.min(output_image)
      output_image = ((output_image - minv) / (maxv - minv) * 255).astype(np.uint8)
      imsave(FLAGS.output, output_image)

def main(unused_argv):
  inference()

if __name__ == '__main__':
  tf.app.run()

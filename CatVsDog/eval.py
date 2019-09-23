from PIL import Image
import tensorflow as tf
import matplotlib.pylab as plt
import numpy as np
import os
from model import *


test_dir = r'.\test'

def evaluate_test_image(nums):
    '''测试集
    nums: 测试数量
    '''
    image_list_test = get_image(test_dir, nums)

    n_class = 2

    input_quene = tf.train.slice_input_producer([image_list_test])
    image_contents = tf.read_file(input_quene[0])
    image_data = tf.image.decode_jpeg(image_contents, channels=3)
    image_resized = tf.image.resize_image_with_crop_or_pad(image_data, 198, 198)
    image_standard = tf.image.per_image_standardization(image_resized)
    image_standard = tf.reshape(image_standard, [1, 198, 198, 3])
    image_batch = tf.train.batch([image_standard], batch_size=nums)

    logit = model_CIFAR(image_batch, nums, n_class)
    logits_x = tf.nn.softmax(logit)

    init = tf.global_variables_initializer()
    logs_train_dir = r'.\log_train'
    saver = tf.train.Saver()

    with tf.Session() as sess:
        print('Reading checkpoints...')
        sess.run(init)
        ckpt = tf.train.get_checkpoint_state(logs_train_dir)

        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split('\\')[-1].split('-')[-1]
            print(global_step)
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Loaded, global_step {}'.format(global_step))
        else:
            print('No file')

        prediction = sess.run(logits_x)
        max_index = np.argmax(prediction)
        print(max_index)

        # if max_index == 0:
        #     print('This is a cat with possibility %.3f'%prediction[:, 0])
        # else:
        #     print('This is a dog with possibility %.3f'%prediction[:, 1])


def get_image(test_dir, x):
    image_list = os.listdir(test_dir)

    n = len(image_list)
    image_rand = []
    for _ in range(x):
        id_i = np.random.randint(0, n)
        image_to_test = os.path.join(test_dir, image_list[id_i])
        image = Image.open(image_to_test)
        image = image.resize([240, 240])
        image = np.array(image)
        image_rand.append(image)

    # plt.imshow(image)
    # plt.show()

    return image_to_test



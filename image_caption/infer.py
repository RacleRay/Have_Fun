import tensorflow as tf
import numpy as np
import pickle
import cv2
import matplotlib.pyplot as plt
from imageio import imread

from utils import ids_to_words
from model import *


batch_size = 1
maxlen = 20
image_size = 224
hidden_size = 1024
embedding_size = 512
is_training = False

beam_width = 10
image_path = 'test.jpg'

MEAN_VALUES = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 3))

with open('dictionary.pkl', 'rb') as fr:
    [vocabulary, word2id, id2word] = pickle.load(fr)


def pre_picture(path):
    img = imread(path)
    if img.shape[-1] == 4:
        img = img[:, :, :-1]
    h = img.shape[0]
    w = img.shape[1]

    if h > w:
        img = img[h // 2 - w // 2: h // 2 + w // 2, :]
    else:
        img = img[:, w // 2 - h // 2: w // 2 + h // 2]

    img = cv2.resize(img, (image_size, image_size))
    X_data = np.expand_dims(img, 0)

    return X_data, img


def show_result(img, sentence, alphas):

    img = (img - img.min()) / (img.max() - img.min())
    # 每个词会显示一张图片
    n = int(np.ceil(np.sqrt(len(sentence))))

    plt.figure(figsize=(10, 8))
    for i in range(len(sentence)):
        word = sentence[i]

        att_weight = np.reshape(alphas[i], (14, 14))
        att_weight = cv2.resize(att_weight, (image_size, image_size))
        att_weight = np.expand_dims(att_weight, -1)
        att_weight = (att_weight - att_weight.min()) / (att_weight.max() - att_weight.min())
        combine = 0.5 * img +  0.5 * att_weight

        plt.subplot(n, n, i + 1)
        plt.text(0, 1, word, color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(combine)
        plt.axis('off')

    plt.show()


def main():
    X = tf.placeholder(tf.float32, [None, image_size, image_size, 3])
    encoded = vgg_endpoints(X - MEAN_VALUES)['conv5_3']
    num_block = encoded.shape[1] * encoded.shape[2]
    num_filter = encoded.shape[3]

    res_op = beam_search_decode(encoded,
                                word2id['<pad>'],
                                len(word2id),
                                maxlen,
                                num_block,
                                num_filter,
                                hidden_size,
                                embedding_size,
                                is_training)

    # beam search 使用到的图节点，解释见beam_search_decode()函数
    initial_state   = res_op[0]
    initial_memory  = res_op[1]
    contexts_placeh = res_op[2]
    last_memory     = res_op[3]
    last_state      = res_op[4]
    last_word       = res_op[5]
    contexts        = res_op[6]
    current_memory  = res_op[7]
    current_state   = res_op[8]
    probs           = res_op[9]
    alpha           = res_op[10]

    # restore
    MODEL_DIR = 'model'
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(MODEL_DIR))

    X_data, img = pre_picture(image_path)

    # 只处理一张图片的beam search，注释见 eval.py
    contexts_, initial_memory_, initial_state_ = sess.run([contexts, initial_memory, initial_state],
                                                            feed_dict={X: X_data})

    result = [{
        'sentence': [],
        'memory': initial_memory_[0],
        'state': initial_state_[0],
        'score': 1.0,
        'alphas': []
    }]
    complete = []
    for t in range(maxlen + 1):
        cache = []
        step = 1 if t == 0 else beam_width
        for s in range(step):
            if t == 0:
                last_word_ = np.ones([batch_size], np.int32) * word2id['<start>']
            else:
                last_word_ = np.array([result[s]['sentence'][-1]], np.int32)

            last_memory_ = np.array([result[s]['memory']], np.float32)
            last_state_ = np.array([result[s]['state']], np.float32)

            current_memory_, current_state_, probs_, alpha_ = sess.run(
                [current_memory, current_state, probs, alpha], feed_dict={
                    contexts_placeh: contexts_,
                    last_memory: last_memory_,
                    last_state: last_state_,
                    last_word: last_word_
                    })

            word_and_probs = [[w, p] for w, p in enumerate(probs_[0])]
            word_and_probs.sort(key=lambda x:-x[1])
            word_and_probs = word_and_probs[:beam_width + 1]

            for w, p in word_and_probs:
                item = {
                    'sentence': result[s]['sentence'] + [w],
                    'memory': current_memory_[0],
                    'state': current_state_[0],
                    'score': result[s]['score'] * p,
                    'alphas': result[s]['alphas'] + [alpha_[0]]
                }
                if id2word[w] == '<end>':
                    complete.append(item)
                else:
                    cache.append(item)

        cache.sort(key=lambda x:-x['score'])
        cache = cache[:beam_width]
        result = cache.copy()


    # 输出预测 sentence 和 attention weight
    if len(complete) == 0:
        final_sentence = result[0]['sentence']
        alphas = result[0]['alphas']
    else:
        final_sentence = complete[0]['sentence']
        alphas = complete[0]['alphas']


    sentence = ids_to_words(final_sentence, id2word)
    print('预测结果为：', sentence)
    sentence = sentence.split(' ')

    print('attention weight可视化')
    show_result(img, sentence, alphas)


if __name__ == '__main__':
    main()


import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from utils import *
from model import *


# parameters
batch_size = 128
maxlen = 20
hidden_size = 1024
embedding_size = 512
image_size = 224
is_training = True
# 借鉴google net的处理，来自大量数据的像素均值
MEAN_VALUES = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 3))


# train data
train_json = 'data/train/captions_train2014.json'
train_ids, train_captions, train_dict = load_data(
    'data/train/images/COCO_train2014_', train_json, maxlen, image_size)

# load captions text vocab
id2word, word2id = word_map(train_captions)
# captions to ids
train_captions = words_to_ids(train_captions, word2id, maxlen)


def main():
    # model input
    X = tf.placeholder(tf.float32, [None, image_size, image_size, 3])
    Y = tf.placeholder(tf.int32, [None, maxlen + 2])

    encoded = vgg_endpoints(X - MEAN_VALUES)['conv5_3']
    num_block = encoded.shape[1] * encoded.shape[2]
    num_filter = encoded.shape[3]
    # print(encoded)  # shape=(?, 14, 14, 512)

    # 网络定义
    alphas, seq_loss, logits = seq_decode(encoded,
                                        Y,
                                        word2id['<pad>'],
                                        len(word2id),
                                        maxlen, num_block,
                                        num_filter,
                                        hidden_size,
                                        embedding_size,
                                        is_training)

    train_op, attention_loss, total_loss = loss_op(alphas, seq_loss, batch_size, maxlen, num_block)

    # 训练模型
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    OUTPUT_DIR = 'model'
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    tf.summary.scalar('losses/loss', seq_loss)
    tf.summary.scalar('losses/attention_loss', attention_loss)
    tf.summary.scalar('losses/total_loss', total_loss)
    summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(OUTPUT_DIR)

    epochs = 20
    for e in range(epochs):
        train_ids, train_captions = shuffle(train_ids, train_captions)
        for i in tqdm(range(len(train_ids) // batch_size)):
            X_batch = np.array([train_dict[x] for x in train_ids[i * batch_size: i * batch_size + batch_size]])
            Y_batch = train_captions[i * batch_size: i * batch_size + batch_size]

            _ = sess.run(train_op, feed_dict={X: X_batch, Y: Y_batch})

            if i > 0 and i % 100 == 0:
                writer.add_summary(sess.run(summary,
                                            feed_dict={X: X_batch, Y: Y_batch}),
                                            e * len(train_ids) // batch_size + i)
                writer.flush()

        saver.save(sess, os.path.join(OUTPUT_DIR, 'image_caption'))

if __name__ == '__main__':
    main()
    print('Train finished')
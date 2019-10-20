import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from tqdm import tqdm
from sklearn.utils import shuffle

from model import build_model
from utils import *

VOCAB_PATH_CH = 'data/vocab.ch'
VOCAB_PATH_EN = 'data/vocab.en'

TRAIN_PATH_CH = 'data/train.ch'
TRAIN_PATH_EN = 'data/train.en'

DEV_PATH_CH = 'data/dev.ch'
DEV_PATH_EN = 'data/dev.en'

TEST_PATH_CH = 'data/test.ch'
TEST_PATH_EN = 'data/test.en'

MODE = 'infer'

OUTPUT_DIR_ = 'model_mine'

embedding_size = 512
hidden_size = 512
maxlen_ch = 62
maxlen_en = 62
num_layers = 2
keep_prob = 0.8
epochs = 20

if MODE == 'infer':
    batch_size = 128
else:
    batch_size = 16

# preprocess data
word2id_ch, id2word_ch, word2id_en, id2word_en = word_map(VOCAB_PATH_CH,
                                                          VOCAB_PATH_EN)

if MODE == 'train':
    ch_path = TRAIN_PATH_CH
    en_path = TRAIN_PATH_EN
elif MODE == 'dev':
    ch_path = DEV_PATH_CH
    en_path = DEV_PATH_EN
elif MODE == 'infer':
    ch_path = TEST_PATH_CH
    en_path = TEST_PATH_EN

sentences_ch, sentences_en, lens_ch, lens_en = load_data(ch_path,
                                                         en_path,
                                                         maxlen_ch,
                                                         maxlen_en,
                                                         word2id_ch,
                                                         word2id_en)
def main():
    # run model in any model
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    if MODE == 'train':
        X, X_len, Y, Y_len, learning_rate, loss, optimizer = build_model(batch_size,
                                                                        hidden_size,
                                                                        embedding_size,
                                                                        MODE,
                                                                        num_layers,
                                                                        keep_prob,
                                                                        maxlen_ch,
                                                                        maxlen_en,
                                                                        word2id_ch,
                                                                        word2id_en)

        saver = tf.train.Saver()
        OUTPUT_DIR = OUTPUT_DIR_
        if not os.path.exists(OUTPUT_DIR):
            os.mkdir(OUTPUT_DIR)

        tf.summary.scalar('loss', loss)
        summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter(OUTPUT_DIR)

        for e in range(epochs):
            total_loss = 0
            total_count = 0

            # 手动设计 lr 变化机制
            start_decay = int(epochs * 2 / 3)
            if e <= start_decay:
                lr = 1.0
            else:
                decay = 0.5 ** (int(4 * (e - start_decay) / (epochs - start_decay)))
                lr = 1.0 * decay

            sess.run(tf.assign(learning_rate, lr))

            train_ch, len_train_ch, train_en, len_train_en = shuffle(sentences_ch,
                                                                    lens_ch,
                                                                    sentences_en,
                                                                    lens_en)

            for i in tqdm(range(train_ch.shape[0] // batch_size)):
                X_batch = train_ch[i * batch_size: i * batch_size + batch_size]
                X_len_batch = len_train_ch[i * batch_size: i * batch_size + batch_size]
                Y_batch = train_en[i * batch_size: i * batch_size + batch_size]
                Y_len_batch = len_train_en[i * batch_size: i * batch_size + batch_size]
                # 因为Y_in和Y_out都比整个Y长度小1.
                Y_len_batch = [l - 1 for l in Y_len_batch]

                feed_dict = {X: X_batch, Y: Y_batch, X_len: X_len_batch, Y_len: Y_len_batch}
                _, ls_ = sess.run([optimizer, loss], feed_dict=feed_dict)

                total_loss += ls_ * batch_size
                total_count += np.sum(Y_len_batch)

                if i > 0 and i % 100 == 0:
                    writer.add_summary(sess.run(summary,
                                                feed_dict=feed_dict),
                                                e * train_ch.shape[0] // batch_size + i)
                    writer.flush()

            print('Epoch %d lr %.3f perplexity %.2f' % (e, lr, np.exp(total_loss / total_count)))
            saver.save(sess, os.path.join(OUTPUT_DIR, 'nmt'))


    if MODE == 'dev':
        X, X_len, Y, Y_len, loss = build_model(batch_size,
                                                hidden_size,
                                                embedding_size,
                                                MODE,
                                                num_layers,
                                                keep_prob,
                                                maxlen_ch,
                                                maxlen_en,
                                                word2id_ch,
                                                word2id_en)

        saver = tf.train.Saver()
        OUTPUT_DIR = OUTPUT_DIR_
        saver.restore(sess, tf.train.latest_checkpoint(OUTPUT_DIR))

        total_loss = 0
        total_count = 0
        for i in tqdm(range(sentences_ch.shape[0] // batch_size)):
            X_batch = sentences_ch[i * batch_size: i * batch_size + batch_size]
            X_len_batch = lens_ch[i * batch_size: i * batch_size + batch_size]
            Y_batch = sentences_en[i * batch_size: i * batch_size + batch_size]
            Y_len_batch = lens_en[i * batch_size: i * batch_size + batch_size]
            Y_len_batch = [l - 1 for l in Y_len_batch]

            feed_dict = {X: X_batch, Y: Y_batch, X_len: X_len_batch, Y_len: Y_len_batch}
            ls_ = sess.run(loss, feed_dict=feed_dict)

            total_loss += ls_ * batch_size
            total_count += np.sum(Y_len_batch)

        print('Dev perplexity %.2f' % np.exp(total_loss / total_count))


    if MODE == 'infer':
        X, X_len, Y, Y_len, sample_id = build_model(batch_size,
                                                    hidden_size,
                                                    embedding_size,
                                                    MODE,
                                                    num_layers,
                                                    keep_prob,
                                                    maxlen_ch,
                                                    maxlen_en,
                                                    word2id_ch,
                                                    word2id_en)

        saver = tf.train.Saver()
        OUTPUT_DIR = OUTPUT_DIR_
        saver.restore(sess, tf.train.latest_checkpoint(OUTPUT_DIR))

        def translate(ids):
            words = [id2word_en[i] for i in ids]
            if words[0] == '<s>':
                words = words[1:]
            if '</s>' in words:
                words = words[:words.index('</s>')]
            return ' '.join(words)

        fw = open('output_infer', 'w')
        for i in tqdm(range(sentences_ch.shape[0] // batch_size)):
            X_batch = sentences_ch[i * batch_size: i * batch_size + batch_size]
            X_len_batch = lens_ch[i * batch_size: i * batch_size + batch_size]
            Y_batch = sentences_en[i * batch_size: i * batch_size + batch_size]
            Y_len_batch = lens_en[i * batch_size: i * batch_size + batch_size]
            Y_len_batch = [l - 1 for l in Y_len_batch]

            feed_dict = {X: X_batch, Y: Y_batch, X_len: X_len_batch, Y_len: Y_len_batch}
            ids = sess.run(sample_id, feed_dict=feed_dict)
            ids = np.transpose(ids, (1, 2, 0))
            ids = ids[:, 0, :]

            for j in range(ids.shape[0]):
                sentence = translate(ids[j])
                fw.write(sentence + '\n')
        fw.close()

        # https://github.com/tensorflow/nmt/
        from nmt.utils.evaluation_utils import evaluate

        for metric in ['bleu', 'rouge']:
            score = evaluate(TEST_PATH_EN, 'output_infer', metric)
            print('翻译结果的的得分为：', metric, score / 100)


if __name__ == '__main__':
    main()
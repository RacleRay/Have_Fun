import tensorflow as tf
import numpy as np
import os
import sys
from copy import deepcopy

import argparse
import Pickle as pkl
from utils import *
from config import *
from models import SiameseNN


parser = argparse.ArgumentParser()
parser.add_argument("--train",  help="whether to train", action='store_true')
parser.add_argument("--test",  help="whether to test", action='store_true')
args = parser.parse_args()


def train(train_corpus, config, val_corpus, eval_train_corpus=None):
    iterator = Iterator(train_corpus)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    with tf.Session(config=config.cf) as sess:
        model = SiameseNN(config)
        saver = tf.train.Saver()
        sess.run(tf.initialize_all_variables())
        for epoch in range(config.num_epochs):
            count = 0
            for batch_x in iterator.next(config.batch_size, shuffle=True):
                batch_qids, batch_q, batch_aids, batch_ap, labels = zip(*batch_x)
                batch_q = np.asarray(batch_q)
                batch_ap = np.asarray(batch_ap)
                _, loss = sess.run([model.train_op, model.total_loss],
                                   feed_dict={model.q:batch_q,
                                              model.a:batch_ap,
                                              model.y:labels,
                                              model.keep_prob:config.keep_prob})
                count += 1
                if count % 10 == 0:
                    print('[epoch {}, batch {}]Loss:{}'.format(epoch, count, loss))
            saver.save(sess,'{}/checkpoint'.format(model_path), global_step=epoch)
            if eval_train_corpus is not None:
                train_res = evaluate(sess, model, eval_train_corpus, config)
                print('[train] ' + train_res)
            if val_corpus is not None:
                val_res = evaluate(sess, model, val_corpus, config)
                print('[eval] ' + val_res)


def evaluate(sess, model, corpus, config):
    iterator = Iterator(corpus)
    count = 0
    total_qids = []
    total_aids = []
    total_pred = []
    total_labels = []
    total_loss = 0.
    for batch_x in iterator.next(config.batch_size, shuffle=False):
        batch_qids, batch_q, batch_aids, batch_ap, labels = zip(*batch_x)
        batch_q = np.asarray(batch_q)
        batch_ap = np.asarray(batch_ap)
        q_ap_cosine, loss = sess.run([model.q_a_cosine, model.total_loss],
                           feed_dict={model.q:batch_q,
                                      model.a:batch_ap,
                                      model.y:labels,
                                      model.keep_prob:1.})
        total_loss += loss
        count += 1
        total_qids.append(batch_qids)
        total_aids.append(batch_aids)
        total_pred.append(q_ap_cosine)
        total_labels.append(labels)
    print('Eval loss:{}'.format(total_loss / count))

    total_qids = np.concatenate(total_qids, axis=0)
    total_aids = np.concatenate(total_aids, axis=0)
    total_pred = np.concatenate(total_pred, axis=0)
    total_labels = np.concatenate(total_labels, axis=0)
    MAP, MRR = eval_map_mrr(total_qids, total_aids, total_pred, total_labels)
    return 'MAP:{}, MRR:{}'.format(MAP, MRR)


def test(corpus, config):
    with tf.Session(config=config.cf) as sess:
        model = SiameseNN(config)
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(model_path))
        print('[test] ' + evaluate(sess, model, corpus, config))


def main(args, model_type='cnn'):
    # pretrained embeddings load
    embeddings = build_embedding(embedding_path, word2id)

    # config
    if model_type == 'cnn':
        config = CNNConfig(max(word2id.values()) + 1, embeddings=embeddings)
    elif model_type == 'rnn':
        config = RNNConfig(max(word2id.values()) + 1, embeddings=embeddings)
    max_q_length = config.max_q_length
    max_a_length = config.max_a_length

    # corpus data
    with open(os.path.join(processed_data_path, 'pointwise_corpus.pkl'), 'r') as fr:
        train_corpus, val_corpus, test_corpus = pkl.load(fr)

    # padding
    train_qids, train_q, train_aids, train_ap, train_labels = zip(*train_corpus)
    train_q = padding(train_q, max_q_length)
    train_ap = padding(train_ap, max_a_length)
    train_corpus = zip(train_qids, train_q, train_aids, train_ap, train_labels)

    val_qids, val_q, val_aids, val_ap, labels = zip(*val_corpus)
    val_q = padding(val_q, max_q_length)
    val_ap = padding(val_ap, max_a_length)
    val_corpus = zip(val_qids, val_q, val_aids, val_ap, labels)

    test_qids, test_q, test_aids, test_ap, labels = zip(*test_corpus)
    test_q = padding(test_q, max_q_length)
    test_ap = padding(test_ap, max_a_length)
    test_corpus = zip(test_qids, test_q, test_aids, test_ap, labels)

    # run
    if args.train:
        train(deepcopy(train_corpus), config, val_corpus, deepcopy(train_corpus))
    elif args.test:
        test(test_corpus, config)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # Get ENV
    ENVIRON = os.environ.copy()

    # data path process
    raw_data_path = './data/WikiQA/raw'
    processed_data_path = './data/WikiQA/processed'
    processed_data_pkl = os.path.join(processed_data_path, 'vocab.pkl')
    embedding_path = './data/embedding/glove.6B.300d.txt'
    model_path = 'models'

    if 'GLOVE_EMBEDDING_6B' in ENVIRON:
        embedding_path = ENVIRON['GLOVE_EMBEDDING_6B']

    print("embedding file: %s" % embedding_path)

    if not os.path.exists(processed_data_pkl):
        raise BaseException(
                "data [{processed_data_pkl}] not exist, run preprocess_wiki.py first.")

    with open(processed_data_pkl, 'r') as fr:
        word2id, id2word = pkl.load(fr)

    # run
    main(args)
# -*- coding: utf-8 -*-
import numpy as np
from keras.preprocessing.sequence import pad_sequences



def load_vocab(path):
    """加载词表"""
    with open(path, 'r', encoding='utf-8') as fr:
        vocab = fr.readlines()
        vocab = [w.strip('\n') for w in vocab]
    return vocab


def word_map(vocab_ch_path, vocab_en_path):
    """两种语言的word2id, id2word的map。'data/vocab.ch', 'data/vocab.en'"""
    vocab_ch = load_vocab(vocab_ch_path)
    vocab_en = load_vocab(vocab_ch_path)

    word2id_ch = {w: i for i, w in enumerate(vocab_ch)}
    id2word_ch = {i: w for i, w in enumerate(vocab_ch)}
    word2id_en = {w: i for i, w in enumerate(vocab_en)}
    id2word_en = {i: w for i, w in enumerate(vocab_en)}

    return word2id_ch, id2word_ch, word2id_en, id2word_en


def preprocess(data_path, word2id):
    """将数据集每一句话转化为id序列

    return：sentences--sentences id序列
            lens--每一个序列的长度
            maxlen--数据集中seq最大长度
    """
    with open(data_path, 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
        sentences = [line.strip('\n').split(' ') for line in lines]
        sentences = [[word2id['<s>']] + [word2id[w] for w in sentence] + [word2id['</s>']] for sentence in sentences]

        lens = [len(sentence) for sentence in sentences]

        return sentences, lens


def compute_maxlen(word2id_ch, word2id_en, train_path, dev_path, test_path):
    """计算最大序列长度, 在数据分析时使用。train_path为包含ch中文和en英文的路径列表"""
    train_ch, lens_cn_train = preprocess(train_path[0], word2id_ch)
    train_en, lens_en_train = preprocess(train_path[1], word2id_en)

    dev_ch, lens_cn_dev = preprocess(dev_path[0], word2id_ch)
    dev_en, lens_en_dev = preprocess(dev_path[1], word2id_en)

    test_ch, lens_cn_test = preprocess(test_path[0], word2id_ch)
    test_en, lens_en_test = preprocess(test_path[1], word2id_en)

    maxlen_ch = max([max(lens_cn_train), max(lens_cn_dev), max(lens_cn_test)])
    maxlen_en = max([max(lens_en_train), max(lens_en_dev), max(lens_en_test)])

    return maxlen_ch, maxlen_en


def load_data(path_ch, path_en, maxlen_ch, maxlen_en, word2id_ch, word2id_en):
    """加载数据集"""
    sentences_ch, lens_ch = preprocess(path_ch, word2id_ch)
    sentences_en, lens_en = preprocess(path_en, word2id_en)

    sentences_ch = pad_sequences(sentences_ch,
                                 maxlen=maxlen_ch,
                                 padding='post',
                                 truncating='post',
                                 value=word2id_ch['</s>'])
    sentences_en = pad_sequences(sentences_en,
                                 maxlen=maxlen_en,
                                 padding='post',
                                 truncating='post',
                                 value=word2id_en['</s>'])

    return sentences_ch, sentences_en, lens_ch, lens_en
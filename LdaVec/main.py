# -*- coding:utf-8 -*-
# author: Racle
# project: LdaVec

import random
import sklearn.datasets
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import shuffle

from lda2vec import LDA2VEC
from skipgramDataGen import skipgrams
from config import *


def 
# 每个文档为一行，分词完成的文本
trainset = sklearn.datasets.load_files(container_path = 'PATH', encoding = 'UTF-8')

# 处理输入
bow = CountVectorizer().fit(trainset.data)
transformed = bow.transform(trainset.data)

idx_text_clean, len_idx_text_clean = [], []
for text in transformed:
    splitted = text.nonzero()[1]
    idx_text_clean.append(splitted)

# word2id
dictionary = {i: no for no, i in enumerate(bow.get_feature_names())}
# id2word
reversed_dictionary = {no: i for no, i in enumerate(bow.get_feature_names())}

freqs = transformed.toarray().sum(axis=0).tolist()
doc_ids = np.arange(len(idx_text_clean))

# 生成样本
num_unique_documents = doc_ids.max()
pivot_words, target_words, doc_ids = [], [], []
for i, t in enumerate(idx_text_clean):
    pairs, _ = skipgrams(
        t,
        vocabulary_size=len(dictionary),
        window_size=window_size,
        shuffle=True,
        negative_samples=0,
    )
    for pair in pairs:
        temp_data = pair
        pivot_words.append(temp_data[0])
        target_words.append(temp_data[1])
        doc_ids.append(i)

# shuffle
pivot_words, target_words, doc_ids = shuffle(pivot_words,
                                             target_words,
                                             doc_ids,
                                             random_state=10)
num_unique_documents = len(idx_text_clean)




# -*- coding:utf-8 -*-
# author: Racle
# project: LdaVec

import random


def skipgrams(sequence,
              vocabulary_size,
              window_size=4,
              negative_samples=0,
              shuffle=True,
              categorical=False,
              seed=None):
    couples = []
    labels = []
    # 获取skip gram正样本
    for i, wi in enumerate(sequence):  # wi -- pivot word
        if not wi:
            continue

        window_start = max(0, i - window_size)
        window_end = min(len(sequence), i + window_size + 1)
        for j in range(window_start, window_end):  # wj -- target words in window
            if j != i:
                wj = sequence[j]
                if not wj:
                    continue
                couples.append([wi, wj])
                if categorical:  # categorical 标签 ： [0, 1](正例), [1, 0](负例)
                    labels.append([0, 1])
                else:
                    labels.append(1)
    # 获取skip gram负样本 NOTE：这是skipgram中直接引用的，本算法中不需要
    # 在LDA2Vec中，nce loss使用fixed_unigram_candidate_sampler完成了负采样
    # 使用时，将negative_samples设为 0.
    if negative_samples > 0:
        num_negative_samples = int(len(labels) * negative_samples)
        words = [c[0] for c in couples]
        random.shuffle(words)

        couples += [[
            words[i % len(words)],
            random.randint(1, vocabulary_size - 1)
        ] for i in range(num_negative_samples)]
        if categorical:
            labels += [[1, 0]] * num_negative_samples
        else:
            labels += [0] * num_negative_samples
    # shuffle
    if shuffle:
        if seed is None:
            seed = random.randint(0, 10e6)
        random.seed(seed)
        random.shuffle(couples)
        random.seed(seed)
        random.shuffle(labels)

    return couples, labels

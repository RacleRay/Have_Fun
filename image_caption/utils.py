# http://cocodataset.org/#download COCO2014数据集
# 训练集包括8W多张图片，验证集包括4W多张图片，并且提供了每张图片对应的描述
# 每张图片的描述不止一个，因此训练集一共411593个描述，而验证集一共201489个描述，平均一张图片五个描述

import json
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt

from imageio import imread
from tqdm import tqdm


def load_data(image_dir, annotation_path, maxlen, image_size):
    """加载数据，一个图片id只选择一个标题。
    图片保留中心正方形区域并缩放；标题长度不超过20

    input: image_dir--coco图片路径；
           annotation_path--图片描述存放路径，json格式
    return：ids--图片id
            captions--标题
            image_dict--图片id到图片数据的dict
    """
    with open(annotation_path, 'r') as fr:
        annotation = json.load(fr)

    ids = []
    captions = []
    image_dict = {}
    for i in tqdm(range(len(annotation['annotations']))):
        item = annotation['annotations'][i]

        caption = item['caption'].strip().lower()
        caption = caption.replace('.', '').replace(',', '').replace("'", '').replace('"', '')
        caption = caption.replace('&', 'and').replace('(', '').replace(
                    ')', '').replace('-', ' ').split()
        caption = [w for w in caption if len(w) > 0]

        if len(caption) <= maxlen:
            if not item['image_id'] in image_dict:
                img = imread(image_dir + '%012d.jpg' % item['image_id'])
                h = img.shape[0]
                w = img.shape[1]
                # 截取正方形中间区域
                if h > w:
                    img = img[h // 2 - w // 2: h // 2 + w // 2, :]
                else:
                    img = img[:, w // 2 - h // 2: w // 2 + h // 2]
                img = cv2.resize(img, (image_size, image_size))

                if len(img.shape) < 3:
                    img = np.expand_dims(img, -1)
                    img = np.concatenate([img, img, img], axis=-1)

                image_dict[item['image_id']] = img

            ids.append(item['image_id'])
            captions.append(caption)

    return ids, captions, image_dict


def check_data(train_ids, train_captions, train_dict):
    """查看训练数据"""
    data_index = np.arange(len(train_ids))
    np.random.shuffle(data_index)
    N = 4
    data_index = data_index[:N]

    plt.figure(figsize=(12, 20))
    for i in range(N):
        caption = train_captions[data_index[i]]
        img = train_dict[train_ids[data_index[i]]]

        plt.subplot(4, 1, i + 1)
        plt.imshow(img)
        plt.title(' '.join(caption))
        plt.axis('off')


def word_map(train_captions):
    """根据caption生成词表

    return: id2word, word2id--对应的转换字典
    """
    vocabulary = {}
    for caption in train_captions:
        for word in caption:
            vocabulary[word] = vocabulary.get(word, 0) + 1

    vocabulary = sorted(vocabulary.items(), key=lambda x:-x[1])
    vocabulary = [w[0] for w in vocabulary]

    word2id = {'<pad>': 0, '<start>': 1, '<end>': 2}
    for i, w in enumerate(vocabulary):
        word2id[w] = i + 3
    id2word = {i: w for w, i in word2id.items()}

    print(len(vocabulary), vocabulary[:20])

    with open('dictionary.pkl', 'wb') as fw:
        pickle.dump([vocabulary, word2id, id2word], fw)

    return id2word, word2id


def ids_to_words(ids, id2word):
    """ids序列转换为words序列

    return: sentence, string
    """
    words = [id2word[i] for i in ids if i >= 3]
    return ' '.join(words) + '.'


def words_to_ids(data, word2id, maxlen):
    """words序列转换为ids序列

    return: ids_array--id序列，nparray数据类型
    """
    result = []
    for caption in data:
        vector = [word2id['<start>']]
        for word in caption:
            if word in word2id:
                vector.append(word2id[word])
        vector.append(word2id['<end>'])
        result.append(vector)

    ids_array = np.zeros((len(data), maxlen + 2), np.int32)
    for i in tqdm(range(len(result))):
        ids_array[i, :len(result[i])] = result[i]

    return ids_array


def load_valid_ground_truth(val_ids, val_captions):
    """加载验证数据，一张图片有5句对应描述"""
    gt = {}

    for i in tqdm(range(len(val_ids))):
        val_id = val_ids[i]

        if not val_id in gt:
            gt[val_id] = []

        gt[val_id].append(' '.join(val_captions[i]) + ' .')

    print("验证集的大小为：", len(gt))

    return gt
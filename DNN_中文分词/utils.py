from keras.utils import np_utils
import numpy as np
import re


# 读取字典
vocab = open('data/msr/msr_training_words.utf8').read().rstrip('\n').split('\n')
vocab = list(''.join(vocab))
stat = {}
for v in vocab:
    stat[v] = stat.get(v, 0) + 1
stat = sorted(stat.items(), key=lambda x:x[1], reverse=True)
vocab = [s[0] for s in stat]
# 5167 个字
# print(len(vocab))
# 映射
char2id = {c : i + 1 for i, c in enumerate(vocab)}
id2char = {i + 1 : c for i, c in enumerate(vocab)}
# s：single, b:begin, m:middle, e:end, x：padding
tags = {'s': 0, 'b': 1, 'm': 2, 'e': 3, 'x': 4}


def load_data(path, maxlen=32):
    """标记序列"""
    data = open(path).read().rstrip('\n')
    # 按标点符号和换行符分隔
    data = re.split('[，。！？、\n]', data)
    print('共有数据 %d 条' % len(data))
    print('平均长度：', np.mean([len(d.replace(' ', '')) for d in data]))

    # 准备数据
    X_data = []
    y_data = []
    for sentence in data:
        sentence = sentence.split(' ')
        X = []
        y = []

        try:
            for s in sentence:
                s = s.strip()
                # 跳过空字符
                if len(s) == 0:
                    continue
                # s
                elif len(s) == 1:
                    X.append(char2id[s])
                    y.append(tags['s'])
                elif len(s) > 1:
                    # b
                    X.append(char2id[s[0]])
                    y.append(tags['b'])
                    # m
                    for i in range(1, len(s) - 1):
                        X.append(char2id[s[i]])
                        y.append(tags['m'])
                    # e
                    X.append(char2id[s[-1]])
                    y.append(tags['e'])

            # 统一长度
            if len(X) > maxlen:
                X = X[:maxlen]
                y = y[:maxlen]
            else:  # 补0
                for i in range(maxlen - len(X)):
                    X.append(0)
                    y.append(tags['x'])
        except:
            continue

        else:
            if len(X) > 0:
                X_data.append(X)
                y_data.append(y)

    X_data = np.array(X_data)
    y_data = np_utils.to_categorical(y_data, 5)

    return X_data, y_data

# X_train, y_train = load_data('data/msr/msr_training.utf8')
# X_test, y_test = load_data('data/msr/msr_test_gold.utf8')
# print('X_train size:', X_train.shape)
# print('y_train size:', y_train.shape)
# print('X_test size:', X_test.shape)
# print('y_test size:', y_test.shape)



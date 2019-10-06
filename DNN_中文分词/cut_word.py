from keras.models import Model, load_model
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
tags = {'s': 0, 'b': 1, 'm': 2, 'e': 3, 'x': 4}

maxlen = 32 # 长于32则截断，短于32则填充0
model = load_model('msr_bilstm.h5')


def viterbi(nodes):
    """nodes: 每个词对应的预测结果，标记概率。简单实现，代码粗糙"""
    # 这里的转移概率，最好从训练集统计求得
    trans = {'be': 0.5, 'bm': 0.5, 'eb': 0.5, 'es': 0.5, 'me': 0.5, 'mm': 0.5, 'sb': 0.5, 'ss': 0.5}
    # 标记序列起点
    paths = {'b': nodes[0]['b'], 's': nodes[0]['s']}
    for l in range(1, len(nodes)):  # l：time step
        paths_ = paths.copy()
        paths = {}
        for i in nodes[l].keys():  # i: 序列观测值，observation
            nows = {}
            for j in paths_.keys():   # j: historical path
                if j[-1] + i in trans.keys():  # trans key
                    # 状态转移方程：n[i] = n[i-1] + trans[i-1, i] * state[i]
                    nows[j + i] = paths_[j] + nodes[l][i] * trans[j[-1] + i]
            # (path, socre)
            nows = sorted(nows.items(), key=lambda x: x[1], reverse=True)
            paths[nows[0][0]] = nows[0][1]

    paths = sorted(paths.items(), key=lambda x: x[1], reverse=True)
    return paths[0][0]


def cut_words(data):
    data = re.split('[，。！？、\n]', data)
    sens = []
    Xs = []
    for sentence in data:
        sen = []
        X = []
        sentence = list(sentence)
        # 转换id序列
        for s in sentence:
            s = s.strip()
            if not s == '' and s in char2id:
                sen.append(s)
                X.append(char2id[s])
        # 截断
        if len(X) > maxlen:
            sen = sen[:maxlen]
            X = X[:maxlen]
        else:  # padding
            for i in range(maxlen - len(X)):
                X.append(0)

        if len(sen) > 0:
            Xs.append(X)
            sens.append(sen)

    Xs = np.array(Xs)
    ys = model.predict(Xs)

    results = ''
    for i in range(ys.shape[0]):
        # 模型预测结果
        nodes = [dict(zip(['s', 'b', 'm', 'e'], d[:4])) for d in ys[i]]
        # 求解最优序列
        ts = viterbi(nodes)
        for x in range(len(sens[i])):
            if ts[x] in ['s', 'e']:  # 单个词，或者结尾处
                results += sens[i][x] + '/'
            else:
                results += sens[i][x]

    return results[:-1]


if __name__ == '__main__':
    cut_words('...')
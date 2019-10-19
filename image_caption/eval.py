import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from model import *

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
# https://github.com/tylin/coco-caption

# parameters
batch_size = 128
maxlen = 20
hidden_size = 1024
embedding_size = 512
image_size = 224
is_training = False
beam_width = 10
# 借鉴google net的处理，来自大量数据的像素均值
MEAN_VALUES = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 3))

# valid data
val_json = 'data/val/captions_val2014.json'
val_ids, val_captions, val_dict = load_data(
    'data/val/images/COCO_val2014_', val_json, maxlen, image_size)

# 图片id去重，然后将val_ids补齐为batch_size的整数倍，影响不大
val_ids = list(set(val_ids))
if len(val_ids) % batch_size != 0:
    for i in range(batch_size - len(val_ids) % batch_size):
        val_ids.append(val_ids[0])

# 验证集正确答案
ground_truth = load_ground_truth(val_ids, val_captions)
# 加载词表
with open('dictionary.pkl', 'rb') as fr:
    [vocabulary, word2id, id2word] = pickle.load(fr)


def main():
    """beam search decode validation."""
    X = tf.placeholder(tf.float32, [None, image_size, image_size, 3])
    encoded = vgg_endpoints(X - MEAN_VALUES)['conv5_3']
    num_block = encoded.shape[1] * encoded.shape[2]
    num_filter = encoded.shape[3]
    # print(encoded)  # shape=(?, 14, 14, 512)

    # model graph
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

    # 储存图片id到解码结果的字典。
    id2sentence = {}
    for i in tqdm(range(len(val_ids) // batch_size)):
        X_batch = np.array([val_dict[x] for x in val_ids[i * batch_size: i * batch_size + batch_size]])
        # 第一步输出结果
        contexts_, initial_memory_, initial_state_ = sess.run([contexts, initial_memory, initial_state],
                                                                feed_dict={X: X_batch})

        # 一个batch输入每个time step的中间结果
        result = []
        complete = []
        for b in range(batch_size):
            result.append([{
                'sentence': [],
                'memory': initial_memory_[b],
                'state': initial_state_[b],
                'score': 1.0,
                'alphas': []
            }])
            complete.append([])

        # beam search
        for t in range(maxlen + 1):
            cache = [[] for b in range(batch_size)]
            step = 1 if t == 0 else beam_width
            # 分别在beam_width分支上，进行计算
            for s in range(step):
                if t == 0:
                    last_word_ = np.ones([batch_size], np.int32) * word2id['<start>']
                else:
                    last_word_ = np.array([result[b][s]['sentence'][-1] \
                                            for b in range(batch_size)], np.int32)

                last_memory_ = np.array([result[b][s]['memory'] \
                                            for b in range(batch_size)], np.float32)
                last_state_ = np.array([result[b][s]['state'] \
                                            for b in range(batch_size)], np.float32)

                # 计算下一步结果
                current_memory_, current_state_, probs_, alpha_ = sess.run(
                    [current_memory, current_state, probs, alpha], feed_dict={
                        contexts_placeh: contexts_,
                        last_memory: last_memory_,
                        last_state: last_state_,
                        last_word: last_word_
                        })

                # 计算当前每个分支中，前beam_width大的probability的words
                for b in range(batch_size):
                    word_and_probs = [[w, p] for w, p in enumerate(probs_[b])]
                    word_and_probs.sort(key=lambda x:-x[1])  # descending
                    # 在每个分支的下一步，搜索beam_width个结果
                    word_and_probs = word_and_probs[:beam_width + 1]

                    for w, p in word_and_probs:
                        item = {
                            'sentence': result[b][s]['sentence'] + [w],
                            'memory': current_memory_[b],
                            'state': current_state_[b],
                            'score': result[b][s]['score'] * p,
                            'alphas': result[b][s]['alphas'] + [alpha_[b]]
                        }

                        # 记录当前step结束的预测结果
                        if id2word[w] == '<end>':
                            complete[b].append(item)
                        # 记录当前step的所有未结束的预测结果
                        else:
                            cache[b].append(item)

            # 在当前分支所有预测结果中，选择前beam_width大的预测结果
            for b in range(batch_size):
                cache[b].sort(key=lambda x:-x['score'])
                cache[b] = cache[b][:beam_width]

            result = cache.copy()

        # 记录最大概率的预测sentence
        for b in range(batch_size):
            if len(complete[b]) == 0:
                final_sentence = result[b][0]['sentence']
            else:
                final_sentence = complete[b][0]['sentence']

            val_id = val_ids[i * batch_size + b]
            if not val_id in id2sentence:
                id2sentence[val_id] = [ids_to_words(final_sentence, id2word)]

    # 保存（预测句子，ground_truth）的结果
    with open('generated.txt', 'w') as fw:
        for i in id2sentence.keys():
            fw.write(str(i) + '^' + id2sentence[i][0] + '^' + '_'.join(ground_truth[i]) + '\n')


    # evaluate
    # 加载（预测句子，ground_truth）
    id2sentence = {}
    gt = {}
    with open('generated.txt', 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            line = line.strip('\n').split('^')
            i = line[0]
            id2sentence[i] = [line[1]]
            gt[i] = line[2].split('_')

    # 评分方法
    scorers = [
        (Bleu(4), ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4']),
        (Rouge(), 'ROUGE_L'),
        (Cider(), 'CIDEr')
    ]

    for scorer, name in scorers:
        score, _ = scorer.compute_score(gt, id2sentence)
        if type(score) == list:
            for n, s in zip(name, score):
                print(n, s)
        else:
            print('Validation result:', name, score)


if __name__ == "__main__":
    main()
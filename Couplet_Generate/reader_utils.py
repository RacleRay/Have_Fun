import random
from queue import Queue
from threading import Thread

def padding_seq(seq):
    """padding每个输入sequence为最大的sequence长度
    arg：seq of ids
    return: results, padding到max_len的id list
    """
    results = []
    max_len = 0
    for s in seq:
        if max_len < len(s):
            max_len = len(s)
    for i in range(0, len(seq)):
        l = max_len - len(seq[i])
        results.append(seq[i] + [0 for j in range(l)])
    return results


def encode_text(words, vocab_indices):
    """把文本序列映射为id序列
    args: words, 输入对联中每个字组成的list
          vocab_indices，词到id的dict
    return：文本序列对应的id序列
    """
    return [vocab_indices[word] for word in words if word in vocab_indices]


def decode_text(labels, vocabs, end_token = '</s>'):
    """把id序列映射为文本序列
    args: labels, decoder输出的预测结果list
          vocab，id到词的dict
    return：results，' '连接的预测文本
    """
    results = []
    for idx in labels:
        word = vocabs[idx]
        if word == end_token:
            return ' '.join(results)
        results.append(word)
    return ' '.join(results)


def read_vocab(vocab_file):
    """读取词表文件
    return：vocabs，list包含文件中的所有字及<s>,</s>,','
    """
    f = open(vocab_file, 'rb')
    vocabs = [line.decode('utf8')[:-1] for line in f]
    f.close()
    return vocabs


class SeqReader():
    """输入序列读取类"""
    def __init__(self, input_file, target_file, vocab_file, batch_size,
            queue_size = 2048, worker_size = 2, end_token = '</s>',
            padding = True, max_len = 50):
        self.input_file = input_file
        self.target_file = target_file
        self.end_token = end_token
        self.batch_size = batch_size
        self.padding = padding
        self.max_len = max_len
        # 读取词汇表
        self.vocabs = read_vocab(vocab_file)
        # 构建词汇与id对应的字典
        self.vocab_indices = dict((c, i) for i, c in enumerate(self.vocabs))
        self.worker_size = worker_size
        self.data_queue = Queue(queue_size)

        with open(self.input_file, 'rb') as f:
            self.single_lines = 1
            for _ in f:
                self.single_lines += 1
        self.data_size = int(self.single_lines / batch_size)
        self.data_pos = 0
        self._init_reader()


    def start(self):
        """多线程运行_init_reader()"""
        for i in range(self.worker_size):
            t = Thread(target=self._init_reader())
            t.daemon = True   # 守护线程，后台运行
            t.start()
        return

    def read_single_data(self):
        """读取一组数据，
        return:{
                'in_seq': in_seq,
                'in_seq_len': len(in_seq),
                'target_seq': target_seq,
                'target_seq_len': len(target_seq) - 1
        }
        """
        if self.data_pos >= len(self.data):
            random.shuffle(self.data)
            self.data_pos = 0
        result = self.data[self.data_pos]
        self.data_pos += 1
        return result

    def read(self):
        """文件读取，预处理数据格式
        self.data保存了转化为id的每组input sequence、target sequence的dict，储
        存在list中
        """
        while True:
            batch = {'in_seq': [],
                    'in_seq_len': [],
                    'target_seq': [],
                    'target_seq_len': []}
            for i in range(0, self.batch_size):
                item = self.read_single_data()
                batch['in_seq'].append(item['in_seq'])
                batch['in_seq_len'].append(item['in_seq_len'])
                batch['target_seq'].append(item['target_seq'])
                batch['target_seq_len'].append(item['target_seq_len'])
            if self.padding:
                batch['in_seq'] = padding_seq(batch['in_seq'])
                batch['target_seq'] = padding_seq(batch['target_seq'])
            yield batch

    def _init_reader(self):
        """文件读取，预处理数据格式
        self.data保存了转化为id的每组input sequence、target sequence的dict，储
        存在list中
        """
        self.data = []
        input_f = open(self.input_file, 'rb')
        target_f = open(self.target_file, 'rb')

        for input_line in input_f:
            input_line = input_line.decode('utf-8')[:-1]
            target_line = target_f.readline().decode('utf-8')[:-1] # target_line按行读取

            input_words = [x for x in input_line.split(' ') if x != '']
            if len(input_words) >= self.max_len:
                input_words = input_words[:self.max_len-1]
            input_words.append(self.end_token)

            target_words = [x for x in target_line.split(' ') if x != '']
            if len(target_words) >= self.max_len:
                target_words = target_words[:self.max_len-1]
            target_words = ['<s>',] + target_words
            target_words.append(self.end_token)

            in_seq = encode_text(input_words, self.vocab_indices)
            target_seq = encode_text(target_words, self.vocab_indices)
            self.data.append({
                'in_seq': in_seq,
                'in_seq_len': len(in_seq),
                'target_seq': target_seq,
                'target_seq_len': len(target_seq) - 1
            })

        self.data_pos = len(self.data)
        input_f.close()
        target_f.close()
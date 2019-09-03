import urllib.request
import collections
import os
import zipfile
import numpy as np
import tensorflow as tf


def maybe_download(filename, url, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
        filename, HTTPMessage = urllib.request.urlretrieve(url + filename, filename)

    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print("Found and verified", filename)
    else:
        print(statinfo.st_size)
        raise Exception( 'Failed to verify ' + filename + '. Can you get to it with a browser?')

    return filename


def read_file(filename):
    """Extract the first file enclosed in a zip file as a list of words."""
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


def build_dataset(words, n_words):
    """Process raw inputs into a dataset.
    return:
        data -- word index list according to input words
        count -- word and count list
        dictionary -- word to index dictionary
        reversed_dictionary -- index to word dictionary
    """
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = {}

    for word, frequency in count:
        dictionary[word] = len(dictionary)

    data = []  # word index list
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)

    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    return data, count, dictionary, reversed_dictionary]


def collect_data(url, filename, expected_bytes, vocabulary_size=10000):
    """download and preprocess"""
    filename = maybe_download(filename, url, expected_bytes)
    vocabulary = read_file(filename)
    print(vocabulary[:7])

    data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,
                                                                vocabulary_size)
    del vocabulary  # Hint to reduce memory.
    return data, count, dictionary, reverse_dictionary


def read_glove_vecs(glove_file):
    with open(glove_file, 'r') as f:
        words = set()
        word_to_vec_map = {}

        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)

    return words, word_to_vec_map

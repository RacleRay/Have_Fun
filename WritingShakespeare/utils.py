from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking
from keras.layers import LSTM
from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import random
import sys
import io


def build_data(text, Tx = 40, stride = 3):
    """
    Create a training set by scanning a window of size Tx over the text corpus, with stride 3.

    Arguments:
    text -- string, corpus of Shakespearian poem
    Tx -- sequence length, number of time-steps (or characters) in one training example
    stride -- how much the window shifts itself while scanning

    Returns:
    X -- list of training examples
    Y -- list of training labels
    """
    X = []
    Y = []

    # stride: 预测间距
    for i in range(len(text) - Tx, stride):
        X.append(text[i:i + Tx])  # 前Tx个词
        Y.append(text[i + Tx])    # 预测第Tx个词

    print('number of training examples:', len(X))

    return X, Y


def vectorization(X, Y, n_x, char_indices, Tx = 40):
    """
    Convert X and Y (lists) into arrays to be given to a recurrent neural network.

    n_x -- len(chars)

    Returns:
    x -- array of shape (m, Tx, len(chars))
    y -- array of shape (m, len(chars))
    """
    m = len(X)
    x = np.zeros((m, Tx, n_x), dtype=np.bool)
    y = np.zeros((m, n_x), dtype=np.bool)

    for i, sentence in enumerate(X):
        for t, char in enumerate(sentence):
            x[i, t, char_indices[char]] = 1
        y[i, char_indices[char]] = 1

    return x, y


def sample(preds, temperature=1.0):
    """helper function to sample an index from a probability array"""
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature  # temperature越大，分布越趋于均匀，结果更多样

    # softmax
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)

    # multinomial
    prob = np.random.multinomial(1, preds, 1)

    out = np.random.choice(range(len(chars)), p = prob.ravel())

    return out


def on_epoch_end(epoch, _):
    """Function invoked at end of each epoch. Prints generated text."""
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(text) - Tx - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + Tx]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x_pred = np.zeros((1, Tx, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()


def generate_output(model, Tx, chars, char_indices, indices_char):
    generated = ''
    # 注意在训练时的输入为：sentence = text[start_index: start_index + Tx]
    usr_input = input("Write the beginning of your poem, the Shakespeare machine will complete it. Your input is: ")
    # zero pad the sentence to Tx characters（length Tx）.
    sentence = ('{0:0>' + str(Tx) + '}').format(usr_input).lower()
    generated += usr_input

    sys.stdout.write("\n\nHere is your poem: \n\n")
    sys.stdout.write(usr_input)

    for i in range(400):

        x_pred = np.zeros((1, Tx, len(chars)))
        # 处理用户输入的sentence
        for t, char in enumerate(sentence):
            if char != '0':
                x_pred[0, t, char_indices[char]] = 1.

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, temperature = 1.0)
        next_char = indices_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

        sys.stdout.write(next_char)
        sys.stdout.flush()

        if next_char == '\n':
            continue
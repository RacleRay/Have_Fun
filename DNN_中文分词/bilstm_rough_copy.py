# -*- coding: utf-8 -*-

from keras.layers import Input, Dense, Embedding, LSTM, Dropout, TimeDistributed, Bidirectional
from keras.models import Model, load_model
from keras.utils import np_utils
import numpy as np
import re
from utils import *


X_train, y_train = load_data('data/msr/msr_training.utf8')
X_test, y_test = load_data('data/msr/msr_test_gold.utf8')


embedding_size = 128
maxlen = 32 # 长于32则截断，短于32则填充0
hidden_size = 64
batch_size = 64
epochs = 50


X = Input(shape=(maxlen,), dtype='int32')
# mask_zero：将padding的‘0’位置mask。
embedding = Embedding(input_dim=len(vocab) + 1, output_dim=embedding_size, input_length=maxlen, mask_zero=True)(X)
blstm = Bidirectional(LSTM(hidden_size, return_sequences=True), merge_mode='concat')(embedding)
blstm = Dropout(0.6)(blstm)
blstm = Bidirectional(LSTM(hidden_size, return_sequences=True), merge_mode='concat')(blstm)
blstm = Dropout(0.6)(blstm)
output = TimeDistributed(Dense(5, activation='softmax'))(blstm)


model = Model(X, output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)
model.save('msr_bilstm.h5')


print(model.evaluate(X_train, y_train, batch_size=batch_size))
print(model.evaluate(X_test, y_test, batch_size=batch_size))
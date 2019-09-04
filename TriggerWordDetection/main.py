import numpy as np
from pydub import AudioSegment
import random
import sys
import io
import os
import glob
from utils import *
from wavDataProcess import *

from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.optimizers import Adam


def model(input_shape):
     """
    Function creating the model's graph in Keras.
    Note that we use a uni-directional RNN rather than a bi-directional RNN.
    This is really important for trigger word detection, because we don`thave
    to wait for the whole 10sec, and detect the trigger word almost immediately
    after it is said.

    Using a spectrogram and optionally a 1D conv layer is a common pre-processing
    step prior to passing audio data to an RNN, GRU or LSTM.

    Argument:
    input_shape -- shape of the model's input data (Tx, n_freq)

    Returns:
    model -- Keras model instance
    """
    X_input = Input(shape = input_shape)

    X = Conv1D(194, 15, stride=4)(X_input)  # ((5511-15)/4 + 1, 194)
    X = BatchNormalization()(X)  # channel wise
    X = Activation('relu')(X)
    X = Dropout(0.8)(X)

    X = GRU(128, return_sequences=True)(X)
    X = Dropout(0.8)(X)
    X = BatchNormalization()(X)

    X = GRU(128, return_sequences=True)(X)
    X = Dropout(0.8)(X)
    X = BatchNormalization()(X)
    X = Dropout(0.8)(X)

    # applies a layer to every temporal slice of an input.
    X = TimeDistributed(Dense(1, activation='sigmoid'))(X)

    model = Model(inputs = X_input, outputs = X)

    return model


def detect_triggerword(filename):
    """Prediction"""
    plt.subplot(2, 1, 1)

    x = graph_spectrogram(filename)
    # the spectogram outputs (freqs, Tx) and we want (Tx, freqs) to input into the model
    x  = x.swapaxes(0,1)
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)

    plt.subplot(2, 1, 2)
    plt.plot(predictions[0,:,0])
    plt.ylabel('probability')
    plt.show()
    return predictions


def chime_on_activate(filename, chime_file, predictions, threshold):
    """
    trigger a "chiming" sound to play when the probability is above a certain threshold.
    """
    audio_clip = AudioSegment.from_wav(filename)
    chime = AudioSegment.from_wav(chime_file)
    Ty = predictions.shape[1]

    # insert a chime sound at most once every 75 output steps
    consecutive_timesteps = 0
    for i in range(Ty):
        consecutive_timesteps += 1

        if predictions[0,i,0] > threshold and consecutive_timesteps > 75:
            audio_clip = audio_clip.overlay(chime,
                            position = ((i / Ty) * audio_clip.duration_seconds) * 1000)
            consecutive_timesteps = 0

    audio_clip.export("chime_output.wav", format='wav')




if __name__ == '__main__':
    # Load preprocessed training examples
    X = np.load("./XY_train/X.npy")
    Y = np.load("./XY_train/Y.npy")

    # Load preprocessed dev set examples
    X_dev = np.load("./XY_dev/X_dev.npy")
    Y_dev = np.load("./XY_dev/Y_dev.npy")

    model = model(input_shape = (Tx, n_freq))
    model = load_model('./models/tr_model.h5')

    opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])

    model.fit(X, Y, batch_size=5, epochs=1)

    loss, acc = model.evaluate(X_dev, Y_dev)  # F1 score or Precision/Recall better
    print("Dev set accuracy = ", acc)


    # prediction and add chime
    filename  = "./raw_data/dev/2.wav"
    chime_file = "audio_examples/chime.wav"
    prediction = detect_triggerword(filename)
    chime_on_activate(filename, chime_file, prediction, 0.5)




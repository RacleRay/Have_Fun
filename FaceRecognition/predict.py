from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from utils import *
from model import *


np.set_printoptions(threshold=np.nan)

# Face Verification: 1:1, 是否是id对应的person
# Face Recognition：1:k，person是否匹配系统中某一个id
# 将image编码为128维向量，使用向量相似来计算。损失函数为triplet loss。


# triplet loss function, aim to:
# - encodings of two images of the same person are quite similar to each other
# - encodings of two images of different persons are very different
def triplet_loss(y_true, y_pred, alpha = 0.2):
    """
    Implementation of the triplet loss

    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)

    Returns:
    loss -- real number, value of the loss
    """
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    margin = tf.add(tf.subtract(pos_dist, neg_dist), alpha)

    loss = tf.reduce_sum(tf.maximum(margin, 0))

    return loss


def verify(image_path, identity, database, model):
    """
    Face verification:
    Function that verifies if the person on the "image_path" image is "identity".

    Arguments:
    image_path -- path to an image
    identity -- string, name of the person you'd like to verify the identity. Has to be a resident of the Happy house.
    database -- python dictionary mapping names of allowed people's names (strings) to their encodings (vectors).
    model -- your Inception model instance in Keras

    Returns:
    dist -- distance between the image_path and the image of "identity" in the database.
    door_open -- True, if the door should open. False otherwise.
    """
    encoding = img_to_encoding(image_path, model)
    # database是来自预训练好的model对identity的编码
    dist = np.linalg.norm((encoding - database[identity]))

    if dist < 0.7:
        print("It's " + str(identity) + ", welcome home!")
        door_open = True
    else:
        print("It's not " + str(identity) + ", please go away")
        door_open = False

    return dist, door_open


def who_is_it(image_path, database, model):
    """
    Face recognition: don`t need identity.
    Implements face recognition for the happy house by finding who is the person on the image_path image.

    Arguments:
    image_path -- path to an image
    database -- database containing image encodings along with the name of the person on the image
    model -- your Inception model instance in Keras

    Returns:
    min_dist -- the minimum distance between image_path encoding and the encodings from the database
    identity -- string, the name prediction for the person on image_path
    """
    encoding = img_to_encoding(image_path, model)

    # find the most similar
    min_dist = 100  # initialize
    for (name, db_enc) in database.items():
        dist = np.linalg.norm((encoding - db_enc))
        if dist < min_dist:
            min_dist = dist
            identity = name

    # check
    if min_dist > 0.7:
        print("Not in the database.")
    else:
        print ("it's " + str(identity) + ", the distance is " + str(min_dist))

    return min_dist, identity


if __name__ == '__main__':
    # Using an ConvNet to compute encodings
    FRmodel = faceRecoModel(input_shape=(3, 96, 96))  # 3743280 params
    FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
    load_weights_from_FaceNet(FRmodel)

    who_is_it("images/camera_0.jpg", database, FRmodel)

    # 效果提升：
    # 1. 每个database中的人使用多张不同状态下的图片，验证时对多张照片同时进行计算
    # 2. 裁剪每个图片，尽量只包含头像信息，使模型更加健壮。
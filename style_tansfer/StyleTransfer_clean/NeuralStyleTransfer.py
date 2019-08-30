import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from utils import *
import numpy as np
import tensorflow as tf


def compute_content_cost(a_C, a_G):
    """
    Computes the content cost, using just a single hidden layer is sufficient.

    Arguments:
    a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G

    Returns:
    J_content -- scalar.
    """
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    a_C_unrolled = tf.transpose(tf.reshape(a_C, [-1, n_C]))
    a_G_unrolled = tf.transpose(tf.reshape(a_G, [-1, n_C]))

    J_content = 1 / (4 * n_C * n_W * n_H) * \
        tf.reduce_sum(tf.square((a_C_unrolled - a_G_unrolled)))


# The value 𝐺𝑖𝑗 measures how similar the activations of filter 𝑖 are to the activations of filter 𝑗.
# the diagonal elements such as 𝐺𝑖𝑖 also measures how active filter 𝑖 is.

# By capturing the prevalence of different types of features (𝐺𝑖𝑖),
# as well as how much different features occur together (𝐺𝑖𝑗), the Style matrix 𝐺 measures the style of an image.
def gram_matrix(A):
    """
    Argument:
    A -- matrix of shape (n_C, n_H*n_W)

    Returns:
    GA -- Gram matrix of A, of shape (n_C, n_C)
    """
    GA = tf.matmul(A, tf.transpose(A))


def compute_layer_style_cost(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G

    Returns:
    J_style_layer -- tensor representing a scalar value
    """
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    a_S = tf.transpose(tf.reshape(a_S, [-1, n_C]))
    a_G = tf.transpose(tf.reshape(a_G, [-1, n_C]))

    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    J_style_layer = 1/(2*n_C*n_H*n_W)**2 * tf.reduce_sum(tf.square((GS - GG)))

    return J_style_layer

# Style Weights
STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]

def compute_style_cost(model, STYLE_LAYERS):
    """
    Computes the overall style cost from several chosen layers

    Arguments:
    model -- our tensorflow model
    STYLE_LAYERS -- A python list containing:
                        - the names of the layers we would like to extract style from
                        - a coefficient for each of them

    Returns:
    J_style -- tensor representing a scalar value
    """
    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:
        out = model[layer_name]

        a_S = sess.run(out)
        a_G = out  # a tensor and hasn't been evaluated yet
                   # will be updated at each iteration.
        J_style_layer = compute_layer_style_cost(a_S, a_G)
        J_style += coeff * J_style_layer

    return J_style


def total_cost(J_content, J_style, alpha = 10, beta = 40):
    """
    Computes the total cost function

    Arguments:
    J_content -- content cost coded above
    J_style -- style cost coded above
    alpha -- hyperparameter weighting the importance of the content cost
    beta -- hyperparameter weighting the importance of the style cost

    Returns:
    J -- total cost as defined by the formula above.
    """
    J = alpha * J_content + beta * J_style
    return J


def model_nn(sess, input_image, num_iterations = 200):
    """
    training process.

    input_image: generate_noise_image.

    return: generated_image
    """
    sess.run(tf.global_variables_initializer())
    sess.run(model['input']).assign(input_image)  # 此时，开始计算a_G

    for i in range(num_iterations):
        sess.run(train_step)
        generated_image = sess.run(model['input'])  # 结果为BP更新后的输入层

        if i%20 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))

            # save current generated image in the "/output" directory
            save_image("output/" + str(i) + ".png", generated_image)

    # save last generated image
    save_image('output/generated_image.jpg', generated_image)

    return generated_image


def run():
    model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")
    # print(model)

    # Reset the graph
    tf.reset_default_graph()
    # Start interactive session
    sess = tf.InteractiveSession()

    content_image = scipy.misc.imread("images/louvre_small.jpg")
    content_image = reshape_and_normalize_image(content_image)

    style_image = scipy.misc.imread("images/monet.jpg")
    style_image = reshape_and_normalize_image(style_image)

    generated_image = generate_noise_image(content_image)

    # Assign the content image to be the input of the VGG model.
    sess.run(model['input'].assign(content_image))
    out = model['conv4_2']
    a_C = sess.run(out)
    a_G = out
    # 单层计算损失
    J_content = compute_content_cost(a_C, a_G)

    # Assign the input of the model to be the "style" image
    sess.run(model['input'].assign(style_image))
    # 多层计算损失
    J_style = compute_style_cost(model, STYLE_LAYERS)

    J = total_cost(J_content, J_style, alpha=10, beta=40)
    optimizer = tf.train.AdamOptimizer(2.0)
    train_step = optimizer.minimize(J)

    generated_image = model_nn(sess, generated_image)

    return generated_image


import tensorflow as tf
import numpy as np
import scipy.io
import os

# vgg19: 作为图像特征提取网络
# LSTM网络：接受vgg19提取特征，输出文本序列

# 预训练model参数
# http://www.vlfeat.org/matconvnet/pretrained/
vgg = scipy.io.loadmat('imagenet-vgg-verydeep-19.mat')
vgg_layers = vgg['layers']

def vgg_endpoints(inputs, reuse=None):
    """定义出vgg19网络，加载预训练参数，计算图像特征

    return: graph--网络及其参数，dict
    """
    with tf.variable_scope('endpoints', reuse=reuse):

        def _weights(layer, expected_layer_name):
            W = vgg_layers[0][layer][0][0][2][0][0]
            b = vgg_layers[0][layer][0][0][2][0][1]
            layer_name = vgg_layers[0][layer][0][0][0][0]
            assert layer_name == expected_layer_name
            return W, b

        def _conv2d_relu(prev_layer, layer, layer_name):
            W, b = _weights(layer, layer_name)
            W = tf.constant(W)
            b = tf.constant(np.reshape(b, (b.size)))
            return tf.nn.relu(tf.nn.conv2d(
                    prev_layer, filter=W, strides=[1, 1, 1, 1], padding='SAME') + b)

        def _avgpool(prev_layer):
            return tf.nn.avg_pool(prev_layer,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='SAME')

        graph = {}
        graph['conv1_1'] = _conv2d_relu(inputs, 0, 'conv1_1')
        graph['conv1_2'] = _conv2d_relu(graph['conv1_1'], 2, 'conv1_2')
        graph['avgpool1'] = _avgpool(graph['conv1_2'])
        graph['conv2_1'] = _conv2d_relu(graph['avgpool1'], 5, 'conv2_1')
        graph['conv2_2'] = _conv2d_relu(graph['conv2_1'], 7, 'conv2_2')
        graph['avgpool2'] = _avgpool(graph['conv2_2'])
        graph['conv3_1'] = _conv2d_relu(graph['avgpool2'], 10, 'conv3_1')
        graph['conv3_2'] = _conv2d_relu(graph['conv3_1'], 12, 'conv3_2')
        graph['conv3_3'] = _conv2d_relu(graph['conv3_2'], 14, 'conv3_3')
        graph['conv3_4'] = _conv2d_relu(graph['conv3_3'], 16, 'conv3_4')
        graph['avgpool3'] = _avgpool(graph['conv3_4'])
        graph['conv4_1'] = _conv2d_relu(graph['avgpool3'], 19, 'conv4_1')
        graph['conv4_2'] = _conv2d_relu(graph['conv4_1'], 21, 'conv4_2')
        graph['conv4_3'] = _conv2d_relu(graph['conv4_2'], 23, 'conv4_3')
        graph['conv4_4'] = _conv2d_relu(graph['conv4_3'], 25, 'conv4_4')
        graph['avgpool4'] = _avgpool(graph['conv4_4'])
        graph['conv5_1'] = _conv2d_relu(graph['avgpool4'], 28, 'conv5_1')
        graph['conv5_2'] = _conv2d_relu(graph['conv5_1'], 30, 'conv5_2')
        graph['conv5_3'] = _conv2d_relu(graph['conv5_2'], 32, 'conv5_3')
        graph['conv5_4'] = _conv2d_relu(graph['conv5_3'], 34, 'conv5_4')
        graph['avgpool5'] = _avgpool(graph['conv5_4'])

        return graph

#############################################################################

# LSTM decoder part
k_initializer = tf.contrib.layers.xavier_initializer()
b_initializer = tf.constant_initializer(0.0)
e_initializer = tf.random_uniform_initializer(-1.0, 1.0)


def dense(inputs, units, activation=tf.nn.tanh, use_bias=True, name=None):
    return tf.layers.dense(inputs,
                        units,
                        activation,
                        use_bias,
                        kernel_initializer=k_initializer,
                        bias_initializer=b_initializer,
                        name=name)


def batch_norm(inputs, name, is_training=True):
    return tf.contrib.layers.batch_norm(inputs,
                                        decay=0.95,
                                        center=True,
                                        scale=True,
                                        is_training=is_training,
                                        updates_collections=None,
                                        scope=name)


def dropout(inputs, is_training=True):
    return tf.layers.dropout(inputs, rate=0.5, training=is_training)


def seq_decode(pic_encode, y_holder, pad_id, vocab_size, maxlen, num_block, num_filter, hidden_size, embedding_size, is_training):
    """比较底层的attention decoder实现，使用BasicLSTMCell。将抽取的vgg特征信息输入，输出文本id序列。

    params：pic_encode--vgg提取的图像feature map，比如'conv5_3'层特征，shape=(?, 14, 14, 512)
            y_holder--Y placeholder, 在训练模式使用
            pad_id--<pad>在word2id中的id
            vocab_size--词表大小
            maxlen--训练数据的最大描述长度
            num_block--为输入特征图的大小，不同block，代表不同感受野，根据pic_encode的shape确定
            num_filter--pic_encode的channel数
            hidden_size--LSTM的hidden size
            embedding_size--LSTM输入文本的embedding维度
            is_training--是否训练模式，影响batch_norm，dropout
    """
    # 处理图片feature map信息, 将feature map的height，width展开在一个维度
    encoded = tf.reshape(pic_encode, [-1, num_block, num_filter])
    contexts = batch_norm(encoded, 'contexts', is_training)  # batch_size, num_block, num_filter

    # --------------------------------------------------------------------------------------------
    # LSTM输入与输出错开一位
    if is_training:
        Y_in = y_holder[:, :-1]
        Y_out = y_holder[:, 1:]
    # mask掉<pad>
    mask = tf.to_float(tf.not_equal(Y_out, pad_id))

    # 将每个feature map的均值构成的tensor映射到hidden_size维度，作为LSTM的hidden state和cell state输入
    with tf.variable_scope('initialize'):
        context_mean = tf.reduce_mean(contexts, 1)
        state = dense(context_mean, hidden_size, name='initial_state')   # batch_size, num_filter
        memory = dense(context_mean, hidden_size, name='initial_memory')

    # 建立embedding层
    with tf.variable_scope('embedding'):
        embeddings = tf.get_variable('weights', [vocab_size, embedding_size],
                                    initializer=e_initializer)
        embedded = tf.nn.embedding_lookup(embeddings, Y_in)

    # 类似conv 1d的效果，在不同feature map之间，增加FC层，学习相互之间的关系
    with tf.variable_scope('projected'):
        projected_contexts = tf.reshape(
            contexts, [-1, num_filter])  # batch_size * num_block, num_filter
        projected_contexts = dense(projected_contexts,
                                num_filter,
                                activation=None,
                                use_bias=False,
                                name='projected_contexts')
        projected_contexts = tf.reshape(
            projected_contexts,
            [-1, num_block, num_filter])  # batch_size, num_block, num_filter

    # --------------------------------------------------------------------------------------------
    # 使用BasicLSTMCell，手动循环，完成模型结构
    lstm = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
    seq_loss = 0
    alphas = []  # attention weight
    for t in range(maxlen + 1):
        # attention机制实现：计算不同block之间的attention。
        with tf.variable_scope('attend'):
            h0 = dense(state, num_filter, activation=None, name='fc_state')         # batch_size, num_filter
            # projected_contexts：混合全图信息；h0：LSTM序列信息
            h0 = tf.nn.relu(projected_contexts + tf.expand_dims(h0, 1))             # batch_size, num_block, num_filter
            h0 = tf.reshape(h0, [-1, num_filter])                                   # batch_size * num_block, num_filter
            h0 = dense(h0, 1, activation=None, use_bias=False, name='fc_attention') # batch_size * num_block, 1
            h0 = tf.reshape(h0, [-1, num_block])                                    # batch_size, num_block
            alpha = tf.nn.softmax(h0)                                               # batch_size, num_block

            # 不同block进行weighted sum
            context = tf.reduce_sum(contexts * tf.expand_dims(alpha, 2), 1, name='context') # batch_size, num_filter
            alphas.append(alpha)

        # 根据当前state，进一步确定attentioned image context，类似一个gate
        with tf.variable_scope('selector'):
            beta = dense(state, 1, activation=tf.nn.sigmoid, name='fc_beta')        # batch_size, 1
            context = tf.multiply(beta, context, name='selected_context')           # batch_size, num_filter

        # 将word embedding与attentioned image context，作为lstm的输入
        with tf.variable_scope('lstm'):
            h0 = tf.concat([embedded[:, t, :], context], 1)                         # batch_size, embedding_size + num_filter
            _, (memory, state) = lstm(inputs=h0, state=[memory, state])

        # 输出处理
        with tf.variable_scope('decode'):
            h0 = dropout(state, is_training)
            h0 = dense(h0, embedding_size, activation=None, name='fc_logits_state')
            # 将context加到输出层，相当于skip connection
            h0 += dense(context, embedding_size, activation=None, use_bias=False, name='fc_logits_context')
            h0 += embedded[:, t, :]
            h0 = tf.nn.tanh(h0)

            h0 = dropout(h0, is_training)
            logits = dense(h0, vocab_size, activation=None, name='fc_logits')

        # mask掉<pad>, 不将<pad>进行BP
        seq_loss += tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y_out[:, t], logits=logits) \
                              * mask[:, t])
        # for loop中循环使用参数
        tf.get_variable_scope().reuse_variables()

        return alphas, seq_loss, logits


def loss_op(alphas, seq_loss, batch_size, maxlen, num_block, optimizer='Adam', gradient_clip=5.0):
    """在预测描述文字损失之上，加入attention

    params: alphas, seq_loss--为seq_decode函数输出
    return: train operation
    """
    alphas = tf.transpose(tf.stack(alphas), (1, 0, 2))  # batch_size, maxlen + 1, num_block
    # 每一个block在不同time step的attention weight之和
    alphas = tf.reduce_sum(alphas, 1)                   # batch_size, num_block
    # 假设每一个time step平局的attention weight为 1 / num_block，那么alphas的期望应该接近(maxlen + 1) / num_block
    # 作为attention_loss的正则约束
    attention_loss = tf.reduce_sum(((maxlen + 1) / num_block - alphas)**2)
    total_loss = (seq_loss + attention_loss) / batch_size

    with tf.variable_scope('optimizer', reuse=tf.AUTO_REUSE):
        global_step = tf.Variable(0, trainable=False)
        # 只训练seq_decode中的参数，startswith('endpoints')是vgg的参数scope
        vars_t = [
            var for var in tf.trainable_variables()
            if not var.name.startswith('endpoints')
        ]
        train_op = tf.contrib.layers.optimize_loss(total_loss,
                                                    global_step,
                                                    0.001,
                                                    optimizer,
                                                    clip_gradients=gradient_clip,
                                                    variables=vars_t)

    return train_op, attention_loss, total_loss


def beam_search_decode(pic_encode, pad_id, vocab_size, maxlen, num_block, num_filter, hidden_size, embedding_size, is_training=False):
    """使用beam search进行解码，比较底层的实现。没有在seq_decode()函数上直接修改，主要是写在一起太乱。

    由于有不同分支的decode，需要每一次输入不同的X[t]，在beam width上进行解码，因此网络输入输出结构稍有不同。

    return: initial_state   --初始hidden state
            initial_memory  --初始cell state
            contexts_placeh --输入下一个time step的contexts placeholder
            last_memory     --输入下一个time step的cell state placeholder
            last_state      --输入下一个time step的hidden state placeholder
            last_word       --输入下一个time step的word idx placeholder
            contexts        --上一步的hidden state与图像attention tensor做weighted sum的结果
            current_memory  --初始图片信息输入
            current_state   --当前输出的hidden state
            probs           --当前输出的word id预测的概率分布
            alpha           --当前输出的attention weight
    """
    encoded = tf.reshape(pic_encode, [-1, num_block, num_filter])
    contexts = batch_norm(encoded, 'contexts', is_training=is_training)

    # node名称一致，加载训练参数
    with tf.variable_scope('initialize'):
        context_mean = tf.reduce_mean(contexts, 1)
        initial_state = dense(context_mean, hidden_size, name='initial_state')
        initial_memory = dense(context_mean, hidden_size, name='initial_memory')

    # 因为不同beam的存在，有多个中间状态，使用placeholder，动态更新
    contexts_placeh = tf.placeholder(tf.float32, [None, num_block, num_filter])
    last_memory = tf.placeholder(tf.float32, [None, hidden_size])
    last_state = tf.placeholder(tf.float32, [None, hidden_size])
    last_word = tf.placeholder(tf.int32, [None])

    with tf.variable_scope('embedding'):
        embeddings = tf.get_variable('weights', [vocab_size, embedding_size], initializer=e_initializer)
        # last_word：一个placeholder，而不是training时的确定输入
        embedded = tf.nn.embedding_lookup(embeddings, last_word)

    with tf.variable_scope('projected'):
        projected_contexts = tf.reshape(contexts_placeh, [-1, num_filter]) # batch_size * num_block, num_filter
        projected_contexts = dense(projected_contexts, num_filter, activation=None, use_bias=False, name='projected_contexts')
        projected_contexts = tf.reshape(projected_contexts, [-1, num_block, num_filter]) # batch_size, num_block, num_filter

    # -------以下网络结构与training一致， 但是每一步的输入，在sess run时计算--------
    lstm = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)

    with tf.variable_scope('attend'):
        h0 = dense(last_state, num_filter, activation=None, name='fc_state')
        h0 = tf.nn.relu(projected_contexts + tf.expand_dims(h0, 1))
        h0 = tf.reshape(h0, [-1, num_filter])
        h0 = dense(h0, 1, activation=None, use_bias=False, name='fc_attention')
        h0 = tf.reshape(h0, [-1, num_block])

        alpha = tf.nn.softmax(h0)
        context = tf.reduce_sum(contexts_placeh * tf.expand_dims(alpha, 2), 1, name='context')

    with tf.variable_scope('selector'):
        beta = dense(last_state, 1, activation=tf.nn.sigmoid, name='fc_beta')
        context = tf.multiply(beta, context, name='selected_context')

    with tf.variable_scope('lstm'):
        h0 = tf.concat([embedded, context], 1)
        _, (current_memory, current_state) = lstm(inputs=h0, state=[last_memory, last_state])

    with tf.variable_scope('decode'):
        h0 = dropout(current_state, is_training=is_training)
        h0 = dense(h0, embedding_size, activation=None, name='fc_logits_state')
        h0 += dense(context, embedding_size, activation=None, use_bias=False, name='fc_logits_context')
        h0 += embedded
        h0 = tf.nn.tanh(h0)
        h0 = dropout(h0, is_training=is_training)

        logits = dense(h0, vocab_size, activation=None, name='fc_logits')
        probs = tf.nn.softmax(logits)

    return initial_state, initial_memory, contexts_placeh, last_memory, last_state, last_word, \
            contexts, current_memory, current_state, probs, alpha
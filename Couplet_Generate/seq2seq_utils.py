import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.layers import core as layers_core


# 设定LSTM的cell类型
def getLayeredCell(layer_size, num_units, input_keep_prob,
        output_keep_prob=1.0):
    return rnn.MultiRNNCell([
        rnn.DropoutWrapper(
            tf.nn.rnn_cell.LSTMCell(name='basic_lstm_cell',num_units=num_units),
            input_keep_prob, output_keep_prob
        ) for i in range(layer_size)])


def bi_encoder(embed_input, in_seq_len, num_units, layer_size, input_keep_prob):
    '''双向rnn编码器
    embed_input：embeddirding后的输入序列
    num_units：隐藏层单元数
    layer_size：rnn层数
    return:
      bidirectional_dynamic_rnn不同于bidirectional_rnn，结果级联分层输出，可concat到一起
      encoder_output: 每个timestep输出，每层输出按最后一维concat到一起
      encoder_state：每层的final state，(output_state_fw, output_state_bw)
    '''
    bi_layer_size = int(layer_size / 2)
    encode_cell_fw = getLayeredCell(bi_layer_size, num_units, input_keep_prob)
    encode_cell_bw = getLayeredCell(bi_layer_size, num_units, input_keep_prob)
    bi_encoder_output, bi_encoder_state = tf.nn.bidirectional_dynamic_rnn(
            cell_fw = encode_cell_fw,
            cell_bw = encode_cell_bw,
            inputs = embed_input,
            sequence_length = in_seq_len,
            dtype = embed_input.dtype,
            time_major = False)

    # concat encode output and state
    encoder_output = tf.concat(bi_encoder_output, -1)
    encoder_state = []
    for layer_id in range(bi_layer_size):
        encoder_state.append(bi_encoder_state[0][layer_id])
        encoder_state.append(bi_encoder_state[1][layer_id])
    encoder_state = tuple(encoder_state)
    return encoder_output, encoder_state


def attention_decoder_cell(encoder_output, in_seq_len, num_units, layer_size,
        input_keep_prob):
    '''attention decoder
    return: 加入attention_mechanim的decoder cell
    '''
    attention_mechanim = tf.contrib.seq2seq.LuongAttention(num_units,
            encoder_output, in_seq_len, scale = True)
    # attention_mechanim = tf.contrib.seq2seq.BahdanauAttention(num_units,
    #         encoder_output, in_seq_len, normalize = True)

    cell = getLayeredCell(layer_size, num_units, input_keep_prob)
    cell = tf.contrib.seq2seq.AttentionWrapper(cell, attention_mechanim,
            attention_layer_size=num_units)
    return cell


# 输出端的全连接层
def decoder_projection(output, output_size):
    return tf.layers.dense(output, output_size, activation=None,
            use_bias=False, name='output_mlp')


# 训练阶段解码器部分
def train_decoder(encoder_output, in_seq_len, target_seq, target_seq_len,
        encoder_state, num_units, layers, embedding, output_size,
        input_keep_prob, projection_layer):
    # 解码结构的cell
    decoder_cell = attention_decoder_cell(encoder_output, in_seq_len, num_units,
            layers, input_keep_prob)
    batch_size = tf.shape(in_seq_len)[0]
    # 初始状态
    init_state = decoder_cell.zero_state(batch_size, tf.float32).clone(
            cell_state=encoder_state)
    # 训练器
    helper = tf.contrib.seq2seq.TrainingHelper(
                target_seq, target_seq_len, time_major=False)
    # 解码器
    decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper,
            init_state, output_layer=projection_layer)
    # 解码输出
    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
            maximum_iterations=100)
    return outputs.rnn_output


# 预测阶段的解码过程
def infer_decoder(encoder_output, in_seq_len, encoder_state, num_units, layers,
        embedding, output_size, input_keep_prob, projection_layer):
    decoder_cell = attention_decoder_cell(encoder_output, in_seq_len, num_units,
            layers, input_keep_prob)

    batch_size = tf.shape(in_seq_len)[0]
    init_state = decoder_cell.zero_state(batch_size, tf.float32).clone(
            cell_state=encoder_state)

    # BasicDecoder实现，实际使用的是下面的beam search
    """
    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            embedding, tf.fill([batch_size], 0), 1)
    decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper,
            init_state, output_layer=projection_layer)
    """
    # 使用beam search解码
    decoder = tf.contrib.seq2seq.BeamSearchDecoder(
        cell=decoder_cell,
        embedding=embedding,
        start_tokens=tf.fill([batch_size], 0),
        end_token=1,
        initial_state=init_state,
        beam_width=10,
        output_layer=projection_layer,
        length_penalty_weight=1.0)

    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
            maximum_iterations=100)
    return outputs.sample_id


# 序列到序列模型
def seq2seq(in_seq, in_seq_len, target_seq, target_seq_len, vocab_size,
        num_units, layers, dropout):
    """seq2seq模型建立
    return: training -- outputs.rnn_output: rnn输出的概率分布结果
            infering -- outputs.sample_id：输出词id
    """
    in_shape = tf.shape(in_seq)
    batch_size = in_shape[0]

    # 训练开启dropout，预测不开启
    if target_seq != None:
        input_keep_prob = 1 - dropout
    else:
        input_keep_prob = 1

    projection_layer = layers_core.Dense(vocab_size, use_bias=False)

    # 对输入和输出序列做embedding
    with tf.device('/gpu:0'):
        embedding = tf.get_variable(
                name = 'embedding',
                shape = [vocab_size, num_units])
    embed_input = tf.nn.embedding_lookup(embedding, in_seq, name='embed_input')

    # 编码
    encoder_output, encoder_state = bi_encoder(embed_input, in_seq_len,
            num_units, layers, input_keep_prob)

    # 解码
    decoder_cell = attention_decoder_cell(encoder_output, in_seq_len, num_units,
            layers, input_keep_prob)
    batch_size = tf.shape(in_seq_len)[0]
    # decoder初始化，权重初始化，并且将cell state初始化为encoder的final state
    init_state = decoder_cell.zero_state(batch_size, tf.float32).clone(
            cell_state=encoder_state)

    if target_seq != None:
        embed_target = tf.nn.embedding_lookup(embedding, target_seq,
                name='embed_target')
        helper = tf.contrib.seq2seq.TrainingHelper(
                    embed_target, target_seq_len, time_major=False)
    else:
        # 0，1分别应对应句子的起始符id和终止符id
        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                embedding, tf.fill([batch_size], 0), 1)

    decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper,
            init_state, output_layer=projection_layer)
    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
            maximum_iterations=100)

    if target_seq != None:
        return outputs.rnn_output
    else:
        return outputs.sample_id


# 损失函数
def seq_loss(output, target, seq_len):
    target = target[:, 1:]
    cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output,
            labels=target)
    batch_size = tf.shape(target)[0]
    loss_mask = tf.sequence_mask(seq_len, tf.shape(output)[1])
    cost = cost * tf.to_float(loss_mask)
    return tf.reduce_sum(cost) / tf.to_float(batch_size)
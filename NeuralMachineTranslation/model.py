import tensorflow as tf


def single_cell(mode, keep_prob, hidden_size):
    if mode == 'train':
        keep_prob_ = keep_prob
    else:
        keep_prob_ = 1.0
    cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=keep_prob_)
    return cell

def multi_cells(num_layers, mode, keep_prob, hidden_size):
    cells = []
    for i in range(num_layers):
        cell = single_cell(mode, keep_prob, hidden_size)
        cells.append(cell)
    return tf.nn.rnn_cell.MultiRNNCell(cells)


def build_model(batch_size, hidden_size, embedding_size, mode, num_layers, keep_prob,
                maxlen_ch, maxlen_en, word2id_ch, word2id_en, beam_width=10):
    """建立翻译seq2seq翻译模型。

    params：mode--"train","val","infer"
            keep_prob--在"train" mode下设置有效
            beam_width--在"infer" mode下设置有效
            num_layers--encoder层数

    return: 根据不同模式返回node，都需要placehloder节点输入，训练op 或者 result op。
    """
    X = tf.placeholder(tf.int32, [None, maxlen_ch])
    X_len = tf.placeholder(tf.int32, [None])
    Y = tf.placeholder(tf.int32, [None, maxlen_en])
    Y_len = tf.placeholder(tf.int32, [None])

    # 训练模式输入Y，进行decoder的teacher forcing，每一步输入正确答案
    # 注：为了提高泛化能力，提高模型在没有teacher forcing的实际条件下的表现，
    #    可以以一定概率在输入上一步的预测结果。
    Y_in = Y[:, :-1]
    Y_out = Y[:, 1:]

    k_initializer = tf.contrib.layers.xavier_initializer()
    e_initializer = tf.random_uniform_initializer(-1.0, 1.0)

    with tf.variable_scope('embedding_X'):
        embeddings_X = tf.get_variable('weights_X',
                                    [len(word2id_ch), embedding_size],
                                    initializer=e_initializer)
        # batch_size, seq_len, embedding_size
        embedded_X = tf.nn.embedding_lookup(embeddings_X, X)

    with tf.variable_scope('embedding_Y'):
        embeddings_Y = tf.get_variable('weights_Y',
                                    [len(word2id_en), embedding_size],
                                    initializer=e_initializer)
        # batch_size, seq_len, embedding_size
        embedded_Y = tf.nn.embedding_lookup(embeddings_Y, Y_in)

    with tf.variable_scope('encoder'):
        fw_cell = multi_cells(num_layers, mode, keep_prob, hidden_size)
        bw_cell = multi_cells(num_layers, mode, keep_prob, hidden_size)

        bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(fw_cell,
                                                            bw_cell,
                                                            embedded_X,
                                                            dtype=tf.float32,
                                                            sequence_length=X_len)
        # case: num_layers = 2
        # print('=' * 100, '\n', bi_outputs)  # for debugging
        # bi_outputs：(fw shape=(?, 62, 512), bw shape=(?, 62, 512))
        encoder_outputs = tf.concat(bi_outputs, -1)  # shape=(?, 62, 1024)

        # print('=' * 100, '\n', bi_state)  # for debugging
        # (
        #  fw: (
        #           LSTMStateTuple(c= shape=(?, 512), h= shape=(?, 512)),
        #           LSTMStateTuple(c= shape=(?, 512), h= shape=(?, 512))
        #       ),
        #  bw: (
        #           LSTMStateTuple(c= shape=(?, 512), h= shape=(?, 512)),
        #           LSTMStateTuple(c= shape=(?, 512), h= shape=(?, 512))
        #       )
        # )
        encoder_state = []
        for i in range(num_layers):
            encoder_state.append(bi_state[0][i])  # forward
            encoder_state.append(bi_state[1][i])  # backward
        encoder_state = tuple(encoder_state)

        # for debugging
        # print('=' * 100)
        # for i in range(len(encoder_state)):
        #     print(i, encoder_state[i])
        # (fw: LSTMStateTuple(c= shape=(?, 512), h= shape=(?, 512)),
        #  bw: LSTMStateTuple(c= shape=(?, 512), h= shape=(?, 512)),
        #  fw: LSTMStateTuple(c= shape=(?, 512), h= shape=(?, 512)),
        #  bw: LSTMStateTuple(c= shape=(?, 512), h= shape=(?, 512)))


    with tf.variable_scope('decoder'):
        beam_width = beam_width
        memory = encoder_outputs

        # beam search使用tile_batch复制多份，使用BeamSearchDecoder API进行计算
        # infer时使用
        if mode == 'infer':
            memory = tf.contrib.seq2seq.tile_batch(memory, beam_width)
            X_len_ = tf.contrib.seq2seq.tile_batch(X_len, beam_width)
            encoder_state = tf.contrib.seq2seq.tile_batch(encoder_state,
                                                        beam_width)
            bs = batch_size * beam_width
        else:
            bs = batch_size
            X_len_ = X_len

        # multiplicative weight
        attention = tf.contrib.seq2seq.LuongAttention(hidden_size,
                                                      memory,
                                                      X_len_,
                                                      scale=True)
        # additive weight
        # attention = tf.contrib.seq2seq.BahdanauAttention(hidden_size,
        #                                                  memory,
        #                                                  X_len_,
        #                                                  normalize=True)

        # 使用bidirectional rnn
        cell = multi_cells(num_layers * 2, mode, keep_prob, hidden_size)
        cell = tf.contrib.seq2seq.AttentionWrapper(cell,
                                                attention,
                                                hidden_size,
                                                name='attention')

        # 初始state，可沿用encoder输出，也有实现方法不用encoder输出
        decoder_initial_state = cell.zero_state(
            bs, tf.float32).clone(cell_state=encoder_state)

        with tf.variable_scope('projected'):
            output_layer = tf.layers.Dense(len(word2id_en),
                                        use_bias=False,
                                        kernel_initializer=k_initializer)

        if mode == 'infer':
            start = tf.fill([batch_size], word2id_en['<s>'])

            decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell,
                                                           embeddings_Y,
                                                           start,
                                                           word2id_en['</s>'],
                                                           decoder_initial_state,
                                                           beam_width,
                                                           output_layer)

            outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder,
                output_time_major=True,
                maximum_iterations=2 * tf.reduce_max(X_len))

            sample_id = outputs.predicted_ids

        else:
            helper = tf.contrib.seq2seq.TrainingHelper(
                embedded_Y, [maxlen_en - 1 for b in range(batch_size)])

            decoder = tf.contrib.seq2seq.BasicDecoder(cell,
                                                    helper,
                                                    decoder_initial_state,
                                                    output_layer)

            outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder, output_time_major=True)

            logits = outputs.rnn_output
            logits = tf.transpose(logits, (1, 0, 2))  # output_time_major=True
            # print(logits)  # shape=(128, ?, 20003)

    # 计算损失
    if mode != 'infer':
        with tf.variable_scope('loss'):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y_out,
                                                                  logits=logits)
            mask = tf.sequence_mask(Y_len, tf.shape(Y_out)[1], tf.float32)
            loss = tf.reduce_sum(loss * mask) / batch_size

    if mode == 'train':
        learning_rate = tf.Variable(0.0, trainable=False)
        params = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, params), 5.0)
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate).apply_gradients(zip(grads, params))

        return X, X_len, Y, Y_len, learning_rate, loss, optimizer

    # 根据不同模式返回node，都需要placehloder节点输入
    if mode == 'dev':
        return X, X_len, Y, Y_len, loss

    if mode == 'infer':
        return X, X_len, Y, Y_len, sample_id

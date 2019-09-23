import tensorflow as tf


def model_CIFAR(images, batch_size, n_classes):
    '''简单卷积神经网络
    参数：
        images: 图片数据
        batch_size：batch_size
        n_classes：类别数量
    返回：
        logits, 未经softmax
    '''
    with tf.variable_scope('conv1') as scope:
        weights = tf.get_variable('weights',
                                 shape = [3,3,3,16],
                                 dtype = tf.float32,
                                 initializer = tf.truncated_normal_initializer(stddev=0.1,dtype = tf.float32))
        biases = tf.get_variable('biases',
                                shape = [16],
                                dtype = tf.float32,
                                initializer = tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(images, weights, strides=[1,1,1,1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1_out = tf.nn.relu(pre_activation, name = scope.name)

    # pool1 and norm1
    with tf.variable_scope('pooling1_lrn') as scope:
        pool1 = tf.nn.max_pool(conv1_out, ksize=[1,2,2,1], strides=[1,2,2,1],
                              padding='SAME', name='pooling1')
        # LRN层: 对局部神经元的活动创建竞争机制，使得其中响应比较大的值变得相对更大，
        #        并抑制其他反馈较小的神经元，增强了模型的泛化能力
        # norm：
        #      sqr_sum[a, b, c, d] = sum(input[a, b, c, d - depth_radius : d + depth_radius + 1] ** 2)
        #      output = input / (bias + alpha * sqr_sum) ** beta
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm1')

    # conv2
    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable('weights',
                                 shape = [3,3,16,10],
                                 dtype = tf.float32,
                                 initializer = tf.truncated_normal_initializer(stddev=0.1,dtype = tf.float32))
        biases = tf.get_variable('biases',
                                shape = [10],
                                dtype = tf.float32,
                                initializer = tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm1, weights, strides=[1,1,1,1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2_out = tf.nn.relu(pre_activation, name = scope.name)

    # pool2 and norm2
    # 交换lrn_pooling2顺序
    with tf.variable_scope('lrn_pooling2') as scope:
        norm2 = tf.nn.lrn(conv2_out, depth_radius=4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm2')
        pool2 = tf.nn.max_pool(norm2, ksize=[1,2,2,1], strides=[1,2,2,1],
                              padding='SAME', name='pooling2')

    # affine3
    with tf.variable_scope('affine3') as scope:
        # flatten
        reshape = tf.reshape(pool2, shape=[batch_size, -1])  # batch_size行
        dim = reshape.get_shape()[1].value

        weights = tf.get_variable('weights',
                                 shape = [dim, 128],
                                 dtype = tf.float32,
                                 initializer = tf.truncated_normal_initializer(stddev=0.005,dtype = tf.float32))
        biases = tf.get_variable('biases',
                                shape = [128],
                                dtype = tf.float32,
                                initializer = tf.constant_initializer(0.1))
        affine3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

    # affine4
    with tf.variable_scope('affine4') as scope:
        weights = tf.get_variable('weights',
                                 shape = [128, 64],
                                 dtype = tf.float32,
                                 initializer = tf.truncated_normal_initializer(stddev=0.005,dtype = tf.float32))
        biases = tf.get_variable('biases',
                                shape = [64],
                                dtype = tf.float32,
                                initializer = tf.constant_initializer(0.1))
        affine4 = tf.nn.relu(tf.matmul(affine3, weights) + biases, name=scope.name)

    # softmax
    with tf.variable_scope('softmax') as scope:
        weights = tf.get_variable('weights',
                                 shape = [64, n_classes],
                                 dtype = tf.float32,
                                 initializer = tf.truncated_normal_initializer(stddev=0.005,dtype = tf.float32))
        biases = tf.get_variable('biases',
                                shape = [n_classes],
                                dtype = tf.float32,
                                initializer = tf.constant_initializer(0.1))
        softmax_out = tf.nn.relu(tf.matmul(affine4, weights) + biases, name=scope.name)
        # softmax_ = tf.nn.softmax()
    return softmax_out


def losses(logits, labels):
    '''计算损失函数结果
    参数：
        logits：模型输出
        labels：实际类别
    返回：
        loss
    '''
    # 交叉熵损失函数
    # 如果使用sparse：则不用对label（0，1）做one encoding
    label_oh = tf.one_hot(labels, 2)
    with tf.variable_scope('loss') as scope:
        softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label_oh,
                                                                              name='softmax_cross_entropy')
        los = tf.reduce_mean(softmax_cross_entropy, name='loss')
        tf.summary.scalar(scope.name + '/loss', los)
    return los


def training(loss, learning_rate):
    '''训练神经网络
    参数：
        loss：loss
        learning_rate: 学习率
    返回：
        训练操作
    '''
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_operation = optimizer.minimize(loss, global_step=global_step)
    return train_operation


def evaluation(logits, labels):
    '''计算准确率
    参数：
        logits：logits， [batch_size, n_classes]
        labels：labels,  [batch_size], in range(0, n_classes)
    返回：
        accuracy
    '''
    # tf.nn.in_top_k: Says whether the targets are in the top K predictions, K = 1, top 1(==max)
    # predictions, targets, K
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name + '/accuracy', accuracy)
    return accuracy
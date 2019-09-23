import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os


# 训练文件路径
train_dir = r'.\train'

def get_file(file_path):
    '''
    功能：获取数据
    输入：
        file_path：文件路径
    返回：
        文件路径列表 和 文件类别标签列表
    '''
    cats = []
    label_cats = []
    dogs = []
    label_dogs = []

    for file in os.listdir(file_path):
        # Traversing a list containing the names of the files
        name = file.split(sep='.')
        if name[0] == 'cat':
            cats.append(os.path.join(file_path, file))
            label_cats.append(0)
        else:
            dogs.append(os.path.join(file_path, file))
            label_dogs.append(1)

    print('{} dog images, {} cat images'.format(len(label_dogs), len(label_cats)))

    # 随机打乱数据集
    # 堆栈, hstack--纵向列表
    image_list = np.hstack((cats, dogs))
    label_list = np.hstack((label_cats, label_dogs))

    # shuffle按行打乱
    temp = np.array([image_list, label_list])  # (2, 25000)
    temp = temp.transpose()                    # (25000, 2)
    np.random.shuffle(temp)

    # 获取打乱后列表
    image_list = list(temp[:, 0])
    label_list = [int(i) for i in list(temp[:, 1])]

    return image_list, label_list


def get_batch(image, label, image_width, image_height, batch_size, capacity, num_epochs):
    '''
    功能：生成训练批次
    参数：
        image：image_list
        label：label_list
        image_width：设置图片宽度，影响训练图片目标识别
        image_height：设置图片高度，影响训练图片目标识别
        batch_size：批次大小
        capacity：训练队列容量
        num_epochs: 原始文件使用轮次，可理解为训练epoch数
    返回：
        batch_size大小的已处理图片数据 batch_size * [image_width, image_height]
    '''
    # 路径，标签转化格式
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # 生成输入队列
    input_queue = tf.train.slice_input_producer([image, label], num_epochs=num_epochs)

    # 读取图片
    image_contents = tf.read_file(input_queue[0])
    image_data = tf.image.decode_jpeg(image_contents, channels=3)

    # 图片数据处理
    image_resized = tf.image.resize_image_with_crop_or_pad(image_data, image_width, image_height)

    # 数据标准化
    image_standard = tf.image.per_image_standardization(image_resized)

    # 生成批次
    label = input_queue[1]
    # min_after_dequeue: defines how big a buffer we will randomly sample
    #                    from -- bigger means better shuffling but slower start up and more
    #                    memory used.
    # capacity: must be larger than min_after_dequeue and the amount larger
    #           determines the maximum we will prefetch.
    #           Recommend:
    #               min_after_dequeue + (num_threads + a small safety margin) * batch_size
    image_batch, label_batch = tf.train.batch([image_standard, label],
                                                batch_size = batch_size,
                                                num_threads = 32,
                                                capacity = capacity)
    label_batch = tf.reshape(label_batch, [batch_size])  # 化为列

    return image_batch, label_batch



def show_batch_img(train_dir):
    image_list, label_list = get_file(train_dir)
    image_batch, label_batch = get_batch(image_list, label_list, 256, 256, 10, 256, 1)

    with tf.Session() as sess:
        i = 0
        # 监控线程状态，enqueue and dequeue
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            while not coord.should_stop() and i < 1:
                img, label = sess.run([image_batch, label_batch])
                for j in np.arange(10):
                    print('label:{}'.format(label[j]))
                    plt.imshow(img[j,:,:,:])
                    plt.show()
                i += 1

        except tf.errors.OutOfRangeError:
            print('done')

        finally:
            # 停止线程
            coord.request_stop()

        coord.join(threads)
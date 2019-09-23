from utils import *
from model import *
import tensorflow as tf


n_classes = 2
image_width = 198
image_height = 198
batch_size = 6
capacity = 6400
max_step = 10000
num_epochs = 10
learning_rate = 0.0001 # <=0.0001

train_dir = r'.\train'
logs_train_dir = r'.\log_train'


def run_training():
    '''训练模型'''

    # 导入数据
    train, train_label = get_file(train_dir)
    # 生成批次
    train_batch, train_label_batch = get_batch(train, train_label, image_width, image_height,
                                               batch_size, capacity, num_epochs)

    # 定义网络模型
    train_logits = model_CIFAR(train_batch, batch_size, n_classes)
    # 定义损失函数
    train_loss = losses(train_logits, train_label_batch)
    # 定义训练操作
    train_operation = training(train_loss, learning_rate)
    # 定义计算准确率操作
    train_accuracy = evaluation(train_logits, train_label_batch)

    # 计算图及记录定义
    sess = tf.Session()
    summary_operation = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    saver = tf.train.Saver()

    # 根据get_batch得到的数据生成队列，进行多线程训练
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        sess.run(tf.global_variables_initializer(),
                 tf.local_variables_initializer())
        for step in np.arange(max_step):
            if coord.should_stop():
                break

            # tra_loss和train_loss，不要使用相同的名称
            _, tra_loss, tra_acc =  sess.run([train_operation, train_loss, train_accuracy])

            if step % 50 == 0:
                print('Step %d, train loss = %.2f, train accuracy = %.2f'%(step, tra_loss, tra_acc))
                summary_str = sess.run(summary_operation)
                train_writer.add_summary(summary_str, step)

            if step % 2000 == 0 or (step+1) == max_step:
                checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        print('Done, epoch limit reached')

    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()
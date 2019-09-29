import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import os
from model import *


dataset = 'celeba' # CelebA
images = glob.glob(os.path.join(dataset, '*.*'))
print(len(images))


def read_image(path, height, width):
    '''[0, 1]之间的image'''
    image = imread(path)
    h = image.shape[0]
    w = image.shape[1]

    if h > w:
        image = image[h // 2 - w // 2: h // 2 + w // 2, :, :]
    else:
        image = image[:, w // 2 - h // 2: w // 2 + h // 2, :]

    image = imresize(image, (height, width))
    return image / 255.


sess = tf.Session()
sess.run(tf.global_variables_initializer())
z_samples = np.random.uniform(-1.0, 1.0, [batch_size, z_dim]).astype(np.float32)
samples = []
loss = {'d': [], 'g': []}

offset = 0
for i in range(60000):
    n = np.random.uniform(-1.0, 1.0, [batch_size, z_dim]).astype(np.float32)

    offset = (offset + batch_size) % len(images)
    batch = np.array([read_image(img, HEIGHT, WIDTH) for img in images[offset: offset + batch_size]])
    batch = (batch - 0.5) * 2 # [-1, 1]

    d_ls, g_ls = sess.run([loss_d, loss_g], feed_dict={X: batch, noise: n, is_training: True})
    loss['d'].append(d_ls)
    loss['g'].append(g_ls)

    # 生成器的训练难度问题，增加训练次数
    sess.run(optimizer_d, feed_dict={X: batch, noise: n, is_training: True})
    sess.run(optimizer_g, feed_dict={X: batch, noise: n, is_training: True})
    sess.run(optimizer_g, feed_dict={X: batch, noise: n, is_training: True})

    if i % 500 == 0:
        print(i, d_ls, g_ls)
        gen_imgs = sess.run(g, feed_dict={noise: z_samples, is_training: False})
        gen_imgs = (gen_imgs + 1) / 2
        imgs = [img[:, :, :] for img in gen_imgs]
        gen_imgs = montage(imgs)
        plt.axis('off')
        plt.imshow(gen_imgs)
        imsave(os.path.join(OUTPUT_DIR, 'sample_%d.jpg' % i), gen_imgs)
        plt.show()
        samples.append(gen_imgs)

saver = tf.train.Saver()
saver.save(sess, os.path.join(OUTPUT_DIR, 'dcgan_' + dataset), global_step=60000)


plt.plot(loss['d'], label='Discriminator')
plt.plot(loss['g'], label='Generator')
plt.legend(loc='upper right')
plt.savefig(os.path.join(OUTPUT_DIR, 'Loss.png'))
plt.show()

# 生成gif
mimsave(os.path.join(OUTPUT_DIR, 'samples.gif'), samples, fps=10)

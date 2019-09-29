import urllib
import tarfile
import os
import glob
import numpy as np
from imageio import imread, imsave, mimsave



def download():
    # 下载和处理LFW数据
    url = 'http://vis-www.cs.umass.edu/lfw/lfw.tgz'
    # CelebA：http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
    filename = 'lfw.tgz'
    directory = 'lfw_imgs'
    new_dir = 'lfw_new_imgs'

    if not os.path.isdir(new_dir):
        os.mkdir(new_dir)

        if not os.path.isdir(directory):
            if not os.path.isfile(filename):
                urllib.request.urlretrieve(url, filename)
            tar = tarfile.open(filename, 'r:gz')
            tar.extractall(path=directory)
            tar.close()

        count = 0
        for dir_, _, files in os.walk(directory):
            for file_ in files:
                img = imread(os.path.join(dir_, file_))
                imsave(os.path.join(new_dir, '%d.png' % count), img)
                count += 1


def montage(images):
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))
    if len(images.shape) == 4 and images.shape[3] == 3:
        m = np.ones(
            (images.shape[1] * n_plots + n_plots + 1,
             images.shape[2] * n_plots + n_plots + 1, 3)) * 0.5
    elif len(images.shape) == 4 and images.shape[3] == 1:
        m = np.ones(
            (images.shape[1] * n_plots + n_plots + 1,
             images.shape[2] * n_plots + n_plots + 1, 1)) * 0.5
    elif len(images.shape) == 3:
        m = np.ones(
            (images.shape[1] * n_plots + n_plots + 1,
             images.shape[2] * n_plots + n_plots + 1)) * 0.5
    else:
        raise ValueError('Could not parse image shape of {}'.format(images.shape))

    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                m[1 + i + i * img_h:1 + i + (i + 1) * img_h,
                  1 + j + j * img_w:1 + j + (j + 1) * img_w] = this_img

    return m
# -*- coding: UTF-8 -*-
# 这个是一个真实的写入图片信息到tfrecord文件，然后读取并还原图片的例子
# 原文：https://blog.csdn.net/u010358677/article/details/70544241

# 程序运行的时候出现下面的错误，可能是在开始训练前就把数据读完退出了，多次测试没有解决这个问题。但是程序本身的逻辑基本了解。
# 代码import_Data_9 解决了下面这个问题，是输入的图片尺寸问题，应该先resize然后在写入tfrecord文件，现在import_Data_8已经修改正确了。
# 根本问题还是图片尺寸问题，resize然后在写入tfrecord文件
# img.resize((300, 300))这个地方的图片尺寸要和读取的图片尺寸一样才可以否则会出现下面的错误。
# 程序中写入的参数type还有图片尺寸，要和读取出来的type还有尺寸一样才可以。这个要注意
# OutOfRangeError (see above for traceback): RandomShuffleQueue '_2_shuffle_batch/random_shuffle_queue' is closed and has insufficient elements (requested 3, current size 0)
# 	 [[node shuffle_batch (defined at /Users/qiulongquan/keras_sample/Import_Data_8.py:109) ]]


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from PIL import Image  # 注意Image,后面会用到
import scipy.io as sio


def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value=[value]))


root_path = '/Users/qiulongquan/keras_sample/'
tfrecords_filename = root_path + 'tfrecords/train.tfrecords'


def write_to_tfrecord():
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)

    height = 300
    width = 300

    txtfile = root_path + 'txt/train.txt'
    fr = open(txtfile)

    for i in fr.readlines():
        item = i.split()
        img = Image.open(root_path + '/images/train_images/' + item[0])
        img = img.resize((300, 300))
        # img_raw = img.tobytes()  # 将图片转化为二进制格式
        # img = np.float64(misc.imread(root_path + '/images/train_images/' + item[0]))
        img = np.float64(img)
        print("img:", type(img))
        # img: <class 'numpy.ndarray'>
        # maskmat = sio.loadmat(root_path + '/mats/train_mats/' + item[1])
        # mask = np.float64(maskmat['seg_mask'])
        label = int(item[2])
        img_raw = img.tostring()
        # img_raw: <class 'bytes'>
        print("img_raw:", type(img_raw))
        # mask_raw = mask.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'name': _bytes_feature(item[0].encode()),
            'image_raw': _bytes_feature(img_raw),
            # 'mask_raw': _bytes_feature(mask_raw),
            'label': _int64_feature(label)}))

        writer.write(example.SerializeToString())

    writer.close()
    fr.close()


def read_and_decode(filename_queue, random_crop=False, random_clip=False, shuffle_batch=True):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
      serialized_example,
      features={
          'height': tf.FixedLenFeature([], tf.int64),
          'width': tf.FixedLenFeature([], tf.int64),
          'name': tf.FixedLenFeature([], tf.string),
          'image_raw': tf.FixedLenFeature([], tf.string),
          # 'mask_raw': tf.FixedLenFeature([], tf.string),
          'label': tf.FixedLenFeature([], tf.int64)
      })

    image = tf.decode_raw(features['image_raw'], tf.float64)
    image = tf.reshape(image, [300, 300, 3])

    # mask = tf.decode_raw(features['mask_raw'], tf.float64)
    # mask = tf.reshape(mask, [300, 300])
    # mask 是随便填充的，这里 1 是不正确的
    mask = 1

    name = features['name']
    label = features['label']
    width = features['width']
    height = features['height']

#    if random_crop:
#        image = tf.random_crop(image, [227, 227, 3])
#    else:
#        image = tf.image.resize_image_with_crop_or_pad(image, 227, 227)

#    if random_clip:
#        image = tf.image.random_flip_left_right(image)

    if shuffle_batch:
        print("shuffle_batch:", shuffle_batch)
        images, names, labels, widths, heights = tf.train.shuffle_batch([image, name, label, width, height],
                                                batch_size=4,
                                                capacity=8000,
                                                num_threads=4,
                                                min_after_dequeue=2000)
    else:
        print("shuffle_batch:", shuffle_batch)
        images, masks, names, labels, widths, heights = tf.train.batch([image, mask, name, label, width, height],
                                        batch_size=4,
                                        capacity=8000,
                                        num_threads=4)
    return images, names, labels, widths, heights


def test_run(tfrecord_filename):
    filename_queue = tf.train.string_input_producer([tfrecord_filename], num_epochs=3)
    tf.local_variables_initializer()
    images, names, labels, widths, heights = read_and_decode(filename_queue)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    # meanfile = sio.loadmat(root_path + 'mats/mean300.mat')
    # meanvalue = meanfile['mean']


    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(1):
            imgs, nms, labs, wids, heis = sess.run([images, names, labels, widths, heights])
            print('batch' + str(i) + ': ')
            print(type(nms[0]))
            print(type(labs[0]))
            print(type(wids[0]))
            print(type(heis[0]))

            for j in range(4):
                print(str(nms[j]) + ': ' + str(labs[j]) + ' ' + str(wids[j]) + ' ' + str(heis[j]))
                img = np.uint8(imgs[j])
                # msk = np.uint8(msks[j])
                plt.subplot(4,2,j*2+1)
                plt.imshow(img)
                plt.subplot(4,2,j*2+2)
                # plt.imshow(msk, vmin=0, vmax=5)
            plt.show()

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    # 先执行write_to_tfrecord创建images文件夹里面图片的tfrecord文件
    # 然后执行test_run(tfrecords_filename)读取tfrecord文件，并通过plt显示图片

    write_to_tfrecord()
    test_run(tfrecords_filename)
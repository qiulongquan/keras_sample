# -*- coding: UTF-8 -*-

# 参考下面2个文章，学习tfrecord
# https://blog.csdn.net/happyhorizion/article/details/77894055
# https://blog.csdn.net/u010358677/article/details/70544241
# 这个程序是一个例子，他没有读取真是的图片，而是采用随机数字填充了tfrecord文件，所以不能读取还原图片
import tensorflow as tf
import numpy as np
import os

# 写入部分
# =============================================================================#
# write images and label in tfrecord file and read them out
def encode_to_tfrecords(tfrecords_filename, data_num):
  ''' write into tfrecord file '''
  if os.path.exists(tfrecords_filename):
    os.remove(tfrecords_filename)

  writer = tf.python_io.TFRecordWriter('./' + tfrecords_filename)  # 创建.tfrecord文件，准备写入

  for i in range(data_num):
    img_raw = np.random.randint(0, 25, size=(5, 5))
    print(img_raw)
    img_raw = img_raw.tostring()
    example = tf.train.Example(features=tf.train.Features(
      feature={
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[i])),
        'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
      }))
    print(example)
    writer.write(example.SerializeToString())

  writer.close()
  return 0

# 读取部分
def decode_from_tfrecords(filename_queue, is_batch):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
  print("serialized_example", serialized_example)
  features = tf.parse_single_example(serialized_example,
                                     features={
                                       'label': tf.FixedLenFeature([], tf.int64),
                                       'img_raw': tf.FixedLenFeature([], tf.string),
                                     })  # 取出包含image和label的feature对象
  image = tf.decode_raw(features['img_raw'], tf.int64)
  image = tf.reshape(image, [5, 5])
  label = tf.cast(features['label'], tf.int64)

  # 而在tensorflow训练时，一般是采取batch的方式去读入数据。tensorflow提供了两种方式，
  # 一种是shuffle_batch（tf.train.shuffle_batch），这种主要是用在训练中，随机选取样本组成batch。
  # 另外一种就是按照数据在tfrecord中的先后顺序生成batch（tf.train.batch）。
  # 这里采用tf.train.shuffle_batch方式：

  if is_batch:
    batch_size = 3
    min_after_dequeue = 10
    capacity = min_after_dequeue + 3 * batch_size
    image, label = tf.train.shuffle_batch([image, label],
                                          batch_size=batch_size,
                                          num_threads=3,
                                          capacity=capacity,
                                          min_after_dequeue=min_after_dequeue)
  return image, label


# =============================================================================#

if __name__ == '__main__':
  # make train.tfrecord
  train_filename = "train.tfrecords"
  encode_to_tfrecords(train_filename, 10)
  ##    # make test.tfrecord
  test_filename = 'test.tfrecords'
  encode_to_tfrecords(test_filename, 3)

  #    run_test = True
  filename_queue = tf.train.string_input_producer([train_filename], num_epochs=None)  # 读入流中
  train_image, train_label = decode_from_tfrecords(filename_queue, is_batch=True)

  filename_queue = tf.train.string_input_producer([test_filename], num_epochs=None)  # 读入流中
  test_image, test_label = decode_from_tfrecords(filename_queue, is_batch=True)
  with tf.Session() as sess:  # 开始一个会话
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    # 下面的这两句代码非常重要，是读取数据必不可少的。
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    try:
      # while not coord.should_stop():
      for i in range(1):
        example, l = sess.run([train_image, train_label])  # 在会话中取出image和label
        print('train:')
        print(example, l)
        texample, tl = sess.run([test_image, test_label])
        print('test:')
        print(texample, tl)
    except tf.errors.OutOfRangeError:
      print('Done reading')
    finally:
      coord.request_stop()

    # 测试函数的最后，要使用以下两句代码进行停止，就如同文件需要close()一样：
    coord.request_stop()
    coord.join(threads)

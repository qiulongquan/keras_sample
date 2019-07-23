import os
import tensorflow as tf

# tensorflow中文档案
# https://www.w3cschool.cn/tensorflow_python/tensorflow_python-h6852fqf.html

# 2个或者多个文件合并在一起处理可以是txt 数组 或者csv 或者其他tfrecord文件

filenames = ["Import_Data_10/file1.txt", "Import_Data_10/file2.txt"]
# dataset = tf.data.TextLineDataset(filenames)

dataset = tf.data.Dataset.from_tensor_slices(filenames)

# Use `Dataset.flat_map()` to transform each file as a separate nested dataset,
# and then concatenate their contents sequentially into a single "flat" dataset.
# * Skip the first line (header row).
# * Filter out lines beginning with "#" (comments).
dataset = dataset.flat_map(
    lambda filename: (
        tf.data.TextLineDataset(filename)
        # 可以跳过第一行用skip
        .skip(1)
        # tf.substr的解释 line字符串中 开始位置0， 长度1的字符获取返回
        # tf.not_equal的解释 获取的第一个字符和#进行比较如果不一样就返回true
        .filter(lambda line: tf.not_equal(tf.substr(line, 0, 1), "#"))))

print(dataset)

# 随机均匀的挑出dataset，创建一个10000大小的缓冲区。这样每次的值就是不一样的了。
dataset = dataset.shuffle(buffer_size=10000)
batched_dataset = dataset.batch(3)

iterator = batched_dataset.make_one_shot_iterator()
next_element = iterator.get_next()

sess = tf.Session()
print(sess.run(next_element))
print(sess.run(next_element))
print(sess.run(next_element))
print("-"*50)

# 输出信息，batch size定义为3，所以一次放入3个样品。一次输出也是3个样品。
# print(sess.run(next_element))一次输出3个样品 一行。
# 最后一行不满3个样品，也可以正常输出。
# [b'ddd eee fff' b'ggggggggggg' b'hhhhhhhhhhh']
# [b'iiiiiiiiiii' b'444 555 666' b'777 777 777']
# [b'888 888 888' b'99999999999']


# 2个数组定义的信息放入一个dataset里面然后batch size定义为4，一次输出一行4个样品
inc_dataset = tf.data.Dataset.range(10)
dec_dataset = tf.data.Dataset.range(0, -10, -1)
dataset = tf.data.Dataset.zip((inc_dataset, dec_dataset))

# 随机均匀的挑出dataset，创建一个10000大小的缓冲区。这样每次的值就是不一样的了。
dataset = dataset.shuffle(buffer_size=10000)
batched_dataset = dataset.batch(4)

iterator = batched_dataset.make_one_shot_iterator()
next_element = iterator.get_next()

sess = tf.Session()
print(sess.run(next_element))  # ==> ([0, 1, 2,   3],   [ 0, -1,  -2,  -3])
print(sess.run(next_element))  # ==> ([4, 5, 6,   7],   [-4, -5,  -6,  -7])
print(sess.run(next_element))  # ==> ([8, 9, 10, 11],   [-8, -9, -10, -11])
print("-"*50)


# 如何通过debug查看dataset里面的值？ tensorflow有一个问题，不知道运行的过程中数据集的值是什么样的。
dataset = tf.data.Dataset.range(100)
dataset = dataset.map(lambda x: tf.fill([tf.cast(x, tf.int32)], x))
# 随机均匀的挑出dataset，创建一个10000大小的缓冲区。这样每次的值就是不一样的了。
dataset = dataset.shuffle(buffer_size=10000)
dataset = dataset.padded_batch(4, padded_shapes=(None,))
# repeat在这里面的作用还不是很清楚，需要继续测试。
dataset = dataset.repeat(10)

iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

print(sess.run(next_element))  # ==> [[0, 0, 0], [1, 0, 0], [2, 2, 0], [3, 3, 3]]
print(sess.run(next_element))  # ==> [[4, 4, 4, 4, 0, 0, 0],
                               #      [5, 5, 5, 5, 5, 0, 0],
                               #      [6, 6, 6, 6, 6, 6, 0],
                               #      [7, 7, 7, 7, 7, 7, 7]]
print(sess.run(next_element))

print("-"*50)
# Output tensor has shape [2, 3].
# fill([2, 3], 9) ==> [[9, 9, 9]
#                      [9, 9, 9]]

# tf.cast就是类型转换
# tensor`a` is [1.8,2.2],dtype = tf.float
# tf.cast(a, tf.int32 ) == >  [ 1 , 2 ]   #dtype = tf.int32

x = 1
print(tf.fill([tf.cast(x, tf.int32), 2], x))

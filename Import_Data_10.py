import os
import tensorflow as tf

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
        .filter(lambda line: tf.not_equal(tf.substr(line, 0, 1), "#"))))

print(dataset)
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
inc_dataset = tf.data.Dataset.range(100)
dec_dataset = tf.data.Dataset.range(0, -100, -1)
dataset = tf.data.Dataset.zip((inc_dataset, dec_dataset))
batched_dataset = dataset.batch(4)

iterator = batched_dataset.make_one_shot_iterator()
next_element = iterator.get_next()

sess = tf.Session()
print(sess.run(next_element))  # ==> ([0, 1, 2,   3],   [ 0, -1,  -2,  -3])
print(sess.run(next_element))  # ==> ([4, 5, 6,   7],   [-4, -5,  -6,  -7])
print(sess.run(next_element))  # ==> ([8, 9, 10, 11],   [-8, -9, -10, -11])
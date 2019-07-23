import os
import tensorflow as tf

# tensorflow中文档案
# https://www.w3cschool.cn/tensorflow_python/tensorflow_python-h6852fqf.html

# 2个或者多个文件合并在一起处理可以是txt 数组 或者csv 或者其他tfrecord文件

filenames = ["Import_Data_10/csv1.csv", "Import_Data_10/csv2.csv"]
# dataset = tf.data.TextLineDataset(filenames)

record_defaults = [tf.float32] * 3   # Eight required float columns
dataset = tf.data.experimental.CsvDataset(filenames, record_defaults)

print("dataset:", dataset)
print("-"*50)
# 随机均匀的挑出dataset，创建一个10000大小的缓冲区。这样每次的值就是不一样的了。
dataset = dataset.shuffle(buffer_size=10000)
batched_dataset = dataset.batch(3)
# repeat在这里面的作用还不是很清楚，需要继续测试。
dataset = dataset.repeat(10)

iterator = batched_dataset.make_one_shot_iterator()
next_element = iterator.get_next()

sess = tf.Session()
# 自动输出所有的元素，到最后弹出raise结束。
while True:
    try:
        print(sess.run(next_element))
    except tf.errors.OutOfRangeError:
        print("raise tf.errors.OutOfRangeError break.")
        break

# for _ in range(100):
#   sess.run(iterator.initializer)
#   while True:
#     try:
#       sess.run(next_element)
#     except tf.errors.OutOfRangeError:
#       print("raise tf.errors.OutOfRangeError break.")
#       break

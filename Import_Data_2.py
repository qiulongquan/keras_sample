import tensorflow as tf
import numpy as np

# tf.random_uniform
# 默然是在0到1之间产生随机数：
# 但是也可以通过maxval指定上界，通过minval指定下界

# tf.random_uniform((6, 6), minval=low,maxval=high,dtype=tf.float32)))
# 返回6*6的矩阵，产生于low和high之间，产生的值是均匀分布的。
with tf.Session() as sess:
    print(sess.run(tf.random_uniform((2, 5), minval=1, maxval=10, dtype=tf.float32)))


sess = tf.Session()
training_dataset = tf.data.Dataset.range(5).map(
    lambda x: x + tf.random_uniform([], 0, 5, tf.int64))

print("completed")

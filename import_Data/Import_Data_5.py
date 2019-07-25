import tensorflow as tf

dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
dataset2 = tf.data.Dataset.from_tensor_slices((tf.random_uniform([4]), tf.random_uniform([4, 100])))
dataset3 = tf.data.Dataset.zip((dataset1, dataset2))

iterator = dataset3.make_initializable_iterator()

sess = tf.Session()
sess.run(iterator.initializer)
next1, (next2, next3) = iterator.get_next()
print(next1.shape)
print(next2)
print(next3)

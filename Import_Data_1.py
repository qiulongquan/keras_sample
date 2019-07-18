from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf


sess = tf.Session()
hello = tf.constant('Hello, TensorFlow!')
print(hello)
print(sess.run(hello))


# Define training and validation datasets with the same structure.
training_dataset = tf.data.Dataset.range(10).map(
    lambda x: x + tf.random_uniform([], 0, 5, tf.int64))
validation_dataset = tf.data.Dataset.range(5)

# A reinitializable iterator is defined by its structure. We could use the
# `output_types` and `output_shapes` properties of either `training_dataset`
# or `validation_dataset` here, because they are compatible.
iterator = tf.data.Iterator.from_structure(training_dataset.output_types,
                                           training_dataset.output_shapes)
next_element = iterator.get_next()

training_init_op = iterator.make_initializer(training_dataset)
validation_init_op = iterator.make_initializer(validation_dataset)


# Run 20 epochs in which the training dataset is traversed, followed by the
# validation dataset.
# 这个地方range指定的数字表示整体运行的次数。
for _ in range(1):
  # Initialize an iterator over the training dataset.
  # 下面的输出结果 sess.run(training_init_op) None
  print("sess.run(training_init_op)", sess.run(training_init_op))
  # 这个range的数字表示多少个元素，和上面的training_dataset定义的range要一致。
  for _ in range(10):
    value = sess.run(next_element)
    print("training ", value)

  # Initialize an iterator over the validation dataset.
  print("sess.run(validation_init_op)", sess.run(validation_init_op))
  for _ in range(5):
    value = sess.run(next_element)
    print("validation ", value)

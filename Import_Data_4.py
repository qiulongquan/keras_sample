import tensorflow as tf

dataset = tf.data.Dataset.range(5)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

# Typically `result` will be the output of a model, or an optimizer's
# training operation.
result = tf.add(next_element, next_element)
# 01234
sess = tf.Session()
sess.run(iterator.initializer)
print(sess.run(result))  # ==> "0"
print(sess.run(result))  # ==> "2"
print(sess.run(result))  # ==> "4"
print(sess.run(result))  # ==> "6"
print(sess.run(result))  # ==> "8"
try:
  sess.run(result)
except tf.errors.OutOfRangeError:
  print("End of dataset")  # ==> "End of dataset"
  # 重新初始化，然后再次运行输出。
  sess.run(iterator.initializer)
  print(sess.run(result))  # ==> "0"
  print(sess.run(result))  # ==> "0"

# 另外一种写法
# sess.run(iterator.initializer)
# while True:
#   try:
#     sess.run(result)
#   except tf.errors.OutOfRangeError:
#     break

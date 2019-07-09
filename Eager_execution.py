from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

tf.enable_eager_execution()
print(tf.executing_eagerly())

print("-"*50)

x = [[2.]]
m = tf.matmul(x, x)
print("hello, {}".format(m))

print("-"*50)

a = tf.constant([[1, 2],
                 [3, 4]])
print(a)

print("-"*50)

# Broadcasting support
b = tf.add(a, 1)
print(b)

print("-"*50)

# Operator overloading is supported
print(a * b)

print("-"*50)
# Use NumPy values
import numpy as np

c = np.multiply(a, b)
print(c)

print("-"*50)
# Obtain numpy value from a tensor:
print(a.numpy())
# => [[1 2]
#     [3 4]]

tfe = tf.contrib.eager

def fizzbuzz(max_num):
  counter = tf.constant(0)
  max_num = tf.convert_to_tensor(max_num)
  for num in range(1, max_num.numpy()+1):
    num = tf.constant(num)
    if int(num % 3) == 0 and int(num % 5) == 0:
      print('FizzBuzz')
    elif int(num % 3) == 0:
      print('Fizz')
    elif int(num % 5) == 0:
      print('Buzz')
    else:
      print(num.numpy())
    counter += 1


fizzbuzz(15)

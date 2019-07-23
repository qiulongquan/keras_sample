import tensorflow as tf
import cv2

# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
def _parse_function(filename, label):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_jpeg(image_string)
  image_resized = tf.image.resize_images(image_decoded, [28, 28])
  return image_resized, label


# Use a custom OpenCV function to read the image, instead of the standard
# TensorFlow `tf.read_file()` operation.
def _read_py_function(filename, label):
  image_decoded = cv2.imread(filename.decode(), cv2.IMREAD_GRAYSCALE)
  return image_decoded, label

# Use standard TensorFlow operations to resize the image to a fixed shape.
def _resize_function(image_decoded, label):
  image_decoded.set_shape([None, None, None])
  image_resized = tf.image.resize_images(image_decoded, [28, 28])
  return image_resized, label


# A vector of filenames.
filenames = tf.constant(["/images/train_images/mmexport1556676578002.jpg",
                         "/images/train_images/mmexport1559434127709.jpg"])

# `labels[i]` is the label for the image in `filenames[i].
labels = tf.constant([1, 2])

dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(
    lambda filename, label: tuple(tf.py_function(
        _read_py_function, [filename, label], [tf.uint8, label.dtype])))
dataset = dataset.map(_resize_function)

# 如何通过debug查看dataset里面的值？ tensorflow有一个问题，不知道运行的过程中数据集的值是什么样的。
print(dataset)
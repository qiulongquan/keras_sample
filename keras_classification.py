#!/usr/bin/env python
# coding: utf-8

# ##### Copyright 2018 The TensorFlow Authors.

# In[78]:


#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# In[79]:


#@title MIT License
#
# Copyright (c) 2017 François Chollet
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


# # 训练您的第一个神经网络: 基本分类

# 本指南训练了一个神经网络模型，来对服装图像进行分类，例如运动鞋和衬衫。如果您不了解所有细节也不需要担心，这是一个对完整TensorFlow项目的简要概述，相关的细节会在需要时进行解释
# 
# 本指南使用[tf.keras](https://www.tensorflow.org/guide/keras)，这是一个用于在TensorFlow中构建和训练模型的高级API。

# In[80]:


from __future__ import absolute_import, division, print_function, unicode_literals

# 导入TensorFlow和tf.keras
import tensorflow as tf
from tensorflow import keras

# 导入辅助库
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)


# ## 导入Fashion MNIST数据集

# 本指南使用[Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) 数据集，其中包含了10个类别中共70,000张灰度图像。图像包含了低分辨率（28 x 28像素）的单个服装物品，如下所示:
# 
# Fashion MNIST 旨在替代传统的[MNIST](http://yann.lecun.com/exdb/mnist/)数据集 — 它经常被作为机器学习在计算机视觉方向的"Hello, World"。MNIST数据集包含手写数字（0,1,2等）的图像，其格式与我们在此处使用的服装相同。
# 
# 本指南使用Fashion MNIST进行多样化，因为它比普通的MNIST更具挑战性。两个数据集都相对较小，用于验证算法是否按预期工作。它们是测试和调试代码的良好起点。
# 
# 我们将使用60,000张图像来训练网络和10,000张图像来评估网络模型学习图像分类任务的准确程度。您可以直接从TensorFlow使用Fashion MNIST，只需导入并加载数据

# In[81]:


fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


# 加载数据集并返回四个NumPy数组:
# 
# * `train_images`和`train_labels`数组是*训练集*—这是模型用来学习的数据。
# * 模型通过*测试集*进行测试, 即`test_images`与 `test_labels`两个数组。
# 
# 图像是28x28 NumPy数组，像素值介于0到255之间。*labels*是一个整数数组，数值介于0到9之间。这对应了图像所代表的服装的*类别*:

# 
# 每个图像都映射到一个标签。由于*类别名称*不包含在数据集中,因此把他们存储在这里以便在绘制图像时使用:

# In[82]:


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# ## 探索数据
# 
# 让我们在训练模型之前探索数据集的格式。以下显示训练集中有60,000个图像，每个图像表示为28 x 28像素:

# In[83]:
# shape函数是numpy.core.fromnumeric中的函数，它的功能是查看矩阵或者数组的维数。
print("train_images.shape", train_images.shape)

# 同样，训练集中有60,000个标签:

# In[84]:

print("len(train_labels)", len(train_labels))

# 每个标签都是0到9之间的整数:

# In[85]:

print("train_labels", train_labels)

# 测试集中有10,000个图像。 同样，每个图像表示为28×28像素:

# In[86]:
print("test_images.shape", test_images.shape)

# 测试集包含10,000个图像标签:

# In[87]:
print("len(test_labels)", len(test_labels))

# ## 数据预处理
# 
# 在训练网络之前必须对数据进行预处理。 如果您检查训练集中的第一个图像，您将看到像素值落在0到255的范围内:

# In[88]:
# train_images里面有60000个图像，显示第1个图像
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# 在馈送到神经网络模型之前，我们将这些值缩放到0到1的范围。为此，我们将像素值值除以255。
# 重要的是，对训练集 和 测试集要以相同的方式进行预处理:

# In[89]:

train_images = train_images / 255.0

test_images = test_images / 255.0

# 显示*训练集*中的前25个图像，并在每个图像下方显示类名。验证数据格式是否正确，我们是否已准备好构建和训练网络。

# In[90]:
# fig1 = plt.figure(num="3*1 inches",figsize=(3,1)) figsize默认是英寸 长3 高1 创建一个画板。
# subplot绘制多图 plt.subplot(5,5,i+1)  就是5*5  25个子图，i+1表示当前是第几个图。
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel("test :"+class_names[train_labels[i]])
plt.show()

# ## 构建模型
# 
# 构建神经网络需要配置模型的层，然后编译模型。

# ### 设置网络层
# 
# 一个神经网络最基本的组成部分便是*网络层*。网络层从提供给他们的数据中提取表示，并期望这些表示对当前的问题更加有意义
# 
# 大多数深度学习是由串连在一起的网络层所组成。大多数网络层，例如`tf.keras.layers.Dense`，具有在训练期间学习的参数。

# In[91]:

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# 网络中的第一层, `tf.keras.layers.Flatten`, 将图像格式从一个二维数组(包含着28x28个像素)转换成为一个包含着28 * 28 = 784个像素的一维数组。
# 可以将这个网络层视为它将图像中未堆叠的像素排列在一起。这个网络层没有需要学习的参数;它仅仅对数据进行格式化。
# 
# 在像素被展平之后，网络由一个包含有两个`tf.keras.layers.Dense`网络层的序列组成。他们被称作稠密链接层或全连接层。
# 第一个`Dense`网络层包含有128个节点(或被称为神经元)。第二个(也是最后一个)网络层是一个包含10个节点的*softmax*层—它将返回包含10个概率分数的数组，总和为1。
# 每个节点包含一个分数，表示当前图像属于10个类别之一的概率。
# 
# ### 编译模型
# 
# 在模型准备好进行训练之前，它还需要一些配置。这些是在模型的*编译(compile)*步骤中添加的:
# 
# * *损失函数* —这可以衡量模型在培训过程中的准确程度。 我们希望将此函数最小化以"驱使"模型朝正确的方向拟合。
# * *优化器* —这就是模型根据它看到的数据及其损失函数进行更新的方式。
# * *评价方式* —用于监控训练和测试步骤。以下示例使用*准确率(accuracy)*，即正确分类的图像的分数。

# In[92]:

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ## 训练模型
# 
# 训练神经网络模型需要以下步骤:
# 
# 1. 将训练数据提供给模型 - 在本案例中，他们是`train_images`和`train_labels`数组。
# 2. 模型学习如何将图像与其标签关联
# 3. 我们使用模型对测试集进行预测, 在本案例中为`test_images`数组。我们验证预测结果是否匹配`test_labels`数组中保存的标签。
# 
# 通过调用`model.fit`方法来训练模型 — 模型对训练数据进行"拟合"。

# In[93]:

model.fit(train_images, train_labels, epochs=1)

# 随着模型训练，将显示损失和准确率等指标。该模型在训练数据上达到约0.88(或88％)的准确度。

# ## 评估准确率
# 
# 接下来，比较模型在测试数据集上的执行情况:

# In[94]:

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)

# 事实证明，测试数据集的准确性略低于训练数据集的准确性。训练精度和测试精度之间的差距是*过拟合*的一个例子。
# 过拟合是指机器学习模型在新数据上的表现比在训练数据上表现更差。

# ## 进行预测
# 
# 通过训练模型，我们可以使用它来预测某些图像。

# In[95]:

predictions = model.predict(test_images)

# 在此，模型已经预测了测试集中每个图像的标签。我们来看看第一个预测:

# In[96]:

print("predictions[0]", predictions[0])

# 预测是10个数字的数组。这些描述了模型的"信心"，即图像对应于10种不同服装中的每一种。我们可以看到哪个标签具有最高的置信度值：

# In[97]:

print("np.argmax(predictions[0]) :", np.argmax(predictions[0]))

# 因此，模型最有信心的是这个图像是ankle boot，或者 `class_names[9]`。 我们可以检查测试标签，看看这是否正确:

# In[98]:

print("test_labels[0]和上面的np.argmax(predictions[0])结果比较一下 看看机器学习预测的是不是正确 :", test_labels[0])

# 我们可以用图表来查看全部10个类别

# In[99]:

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap=plt.cm.binary)
  
  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  # 如果不想显示ticks则可以可以传入空的参数如yticks([])
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)
  
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

# 让我们看看第0个图像，预测和预测数组。

# In[100]:

i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()

# In[101]:

i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()

# 让我们绘制几个图像及其预测结果。正确的预测标签是蓝色的，不正确的预测标签是红色的。该数字给出了预测标签的百分比(满分100)。请注意，即使非常自信，也可能出错。

# In[102]:

# 绘制前X个测试图像，预测标签和真实标签
# 以蓝色显示正确的预测，红色显示不正确的预测
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.show()

# 最后，使用训练的模型对单个图像进行预测。

# In[103]:

# 从测试数据集中获取图像
img = test_images[0]

print(img.shape)

# `tf.keras`模型经过优化，可以一次性对*批量*,或者一个集合的数据进行预测。因此，即使我们使用单个图像，我们也需要将其添加到列表中:

# In[104]:

# 将图像添加到批次中，即使它是唯一的成员。
img = (np.expand_dims(img,0))

print(img.shape)

# 现在来预测图像:

# In[105]:

predictions_single = model.predict(img)

print(predictions_single)

# In[106]:

plot_value_array(0, predictions_single, test_labels)

# 显示x轴的刻标以及对应的标签
# xticks( arange(5), ('Tom', 'Dick', 'Harry', 'Sally', 'Sue') )
# rotation=45这个参数是x轴的lable显示倾斜45度

# 此外xticks()还可以传入matplotlib.text.Text类的属性来控制显示的样式
# https://blog.csdn.net/claroja/article/details/72916695
# 如果不想显示ticks则可以可以传入空的参数如yticks([])
plt.xticks(range(10), class_names, rotation=45)
plt.show()

# `model.predict`返回一个包含列表的列表，每个图像对应一个列表的数据。获取批次中我们(仅有的)图像的预测:

# In[107]:

prediction_result = np.argmax(predictions_single[0])
print(prediction_result)

# 而且，和之前一样，模型预测标签为9。
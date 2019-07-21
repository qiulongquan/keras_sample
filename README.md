```
### 怎样训练一个神经网络: 基本分类  
本指南训练了一个神经网络模型，来对服装图像进行分类，例如运动鞋和衬衫。
这是一个对完整TensorFlow项目的简要概述，相关的细节会进行解释。
本指南使用('https://www.tensorflow.org/guide/keras')，
这是一个用于在TensorFlow中构建和训练模型的高级API。


Keras   分别针对keras里面用到的各个内容进行了说明，比较复杂难懂。
https://www.tensorflow.org/guide/keras
batch_size: NumPyデータを渡されたモデルは、データをバッチに分割し、それを順繰りに舐めて学習を行います。一つのバッチに配分するサンプル数を、バッチサイズとして整数で指定します。全サンプル数がバッチサイズで割り切れない場合、最後のバッチだけ小さくなる可能性があることに注意しましょう。
batch_size应该只是提高处理速度，减少处理时间的，一次放入的样品数量。
一维矩阵到二维矩阵，其实2维矩阵是可以转换为一维矩阵的
a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3]) => [[1. 2. 3.] [4. 5. 6.]]

【TensorFlow】 CPUとGPUの速度比較
https://qiita.com/guitar_char/items/1ff037bd1455a64d6d05
单位的苹果机子第二次开始的训练大概10秒内。 第一次是33秒左右。
--- 訓練完了 ---
かかった時間: 9.4064359664917
对应的github上面的代码
https://github.com/syatyo/tensorflow-practice

Premade Estimator
这个讲解重点是，中间有2个隐藏层，通过和隐藏层进行算法运算，得出不同的可能值，取最大的值。
https://www.tensorflow.org/guide/premade_estimators

Importing Data
To start an input pipeline, you must define a source. For example, to construct a Dataset from some tensors in memory, you can use tf.data.Dataset.from_tensors() or tf.data.Dataset.from_tensor_slices(). Alternatively, if your input data are on disk in the recommended TFRecord format, you can construct a tf.data.TFRecordDataset.

tf.train.shuffle_batch函数输入参数为：
tensor_list: 进入队列的张量列表The list of tensors to enqueue.
batch_size: 从数据队列中抽取一个批次所包含的数据条数The new batch size pulled from the queue.
capacity: 队列中最大的数据条数An integer. The maximum number of elements in the queue.
min_after_dequeue: 提出队列后，队列中剩余的最小数据条数Minimum number elements in the queue after a dequeue, used to ensure a level of mixing of elements.
num_threads: 进行队列操作的线程数目The number of threads enqueuing tensor_list.
seed: 队列中进行随机排列的随机数发生器，似乎不常用到Seed for the random shuffling within the queue.
enqueue_many: 张量列表中的每个张量是否是一个单独的例子，似乎不常用到Whether each tensor in tensor_list is a single example.
shapes: (Optional) The shapes for each example. Defaults to the inferred shapes for tensor_list.
name: (Optional) A name for the operations.
值得注意的是，capacity>=min_after_dequeue+num_threads*batch_size。

通过tf.train.Example写入并读取一个tf图片样本。   OK
https://blog.csdn.net/u010358677/article/details/70544241
https://www.2cto.com/kf/201702/604326.html
已经基本实现了创建tfrecord和读取tfrecord文件的程序。


Multi-Class Neural Networks: Softmax
https://developers.google.cn/machine-learning/crash-course/multi-class-neural-networks/softmax

```
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

Multi-Class Neural Networks: Softmax
https://developers.google.cn/machine-learning/crash-course/multi-class-neural-networks/softmax

```
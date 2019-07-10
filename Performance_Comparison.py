import tensorflow as tf
import input_data
import time

# このコードは重みの初期値が0.01なので、勾配消失の問題がある

# 開始時刻
start_time = time.time() # unixタイム

# MNISTのデータをダウンロードしてローカルへ
print("--- MNISTデータの読み込み開始 ---")
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print("--- MNISTデータの読み込み完了 ---")


def weight(input_unit_size, output_unit_size):
    return tf.Variable(tf.random_normal([input_unit_size, output_unit_size], mean=0.0, stddev=0.01))


def bias(units):
    return tf.Variable(tf.zeros(units))


def affine(input_d, w, b):
    return tf.matmul(input_d, w) + b


def activation(input_d):
    return tf.nn.relu(input_d)


def batch_norm(X, axes, shape, is_training):
    """
    バッチ正規化
    平均と分散による各レイヤの入力を正規化(白色化)する
    """
    if is_training is False:
        return X
    epsilon = 1e-5
    # 平均と分散
    mean, variance = tf.nn.moments(X, axes)
    # scaleとoffsetも学習対象
    scale = tf.Variable(tf.ones([shape]))
    offset = tf.Variable(tf.zeros([shape]))
    return tf.nn.batch_normalization(X, mean, variance, offset, scale, epsilon)


# InputLayer
y_units = 10
x = tf.placeholder(tf.float32, [None, 784])

layer_input = x
hidden_layer_size = 5

w_hists = []
b_hists = []
a_hists = []

for layer in range(1, hidden_layer_size):
    next_layer_units = round((layer_input.shape.as_list()[1] + y_units) * 2 / 3)

    w = weight(layer_input.shape.as_list()[1], next_layer_units)
    b = bias(next_layer_units)

    w_hist = tf.summary.histogram("weight" + str(layer), w)
    b_hist = tf.summary.histogram("bias" + str(layer), b)

    w_hists.append(w_hist)
    b_hists.append(b_hist)

    with tf.name_scope("Wx_b_" + str(layer)):
        z = affine(layer_input, w, b)

    with tf.name_scope("Batch_norm"):
        bn = batch_norm(z, [0], next_layer_units, True)

    with tf.name_scope("Activation_" + str(layer)):
        a = activation(bn)

    a_hist = tf.summary.histogram("activation" + str(layer_input), a)
    a_hists.append(a_hist)

    layer_input = a

w_output_layer = weight(layer_input.shape.as_list()[1], y_units)
b_output_layer = bias(y_units)

with tf.name_scope("Wx_b" + str(hidden_layer_size)):
    z = affine(layer_input, w_output_layer, b_output_layer)

    w_hists.append(tf.summary.histogram("weight" + str(hidden_layer_size + 1), w_output_layer))
    b_hists.append(tf.summary.histogram("bias" + str(hidden_layer_size + 1), b_output_layer))

# name scope を使ってグラフ・ビジュアライザにおけるノードをまとめる。
with tf.name_scope("output_layer"):
    # 出力層でソフトマックス回帰を実行
    y = tf.nn.softmax(z)
    y_hist = tf.summary.histogram("y", y)


# 正解ラベル
label = tf.placeholder(tf.float32, [None, 10], name="label-input")

with tf.name_scope('xent'):
    # 交差エントロピー
    cross_entropy = -tf.reduce_sum(label*tf.log(y))
    tf.summary.scalar("cross entropy", cross_entropy)

# 精度計算
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(label,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

with tf.name_scope("train"):
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)
    acc_summary_train = tf.summary.scalar("acc-train", accuracy)
    loss_summary_train = tf.summary.scalar("cross_entropy_train", cross_entropy)

with tf.name_scope("test"):
    acc_summary_test = tf.summary.scalar("acc-test", accuracy)
    loss_summary_test = tf.summary.scalar("cross_entropy_test", cross_entropy)

with tf.name_scope("val"):
    acc_summary_val = tf.summary.scalar("acc-val", accuracy)
    loss_summary_val = tf.summary.scalar("cross_entropy_val", cross_entropy)


# セッションを作成する。
session = tf.Session()

writer = tf.summary.FileWriter("mnist_logs/five_layer_nn", session.graph_def)

# 変数の初期化
init_op = tf.global_variables_initializer()

# 変数の初期化セッションを実行する
session.run(init_op)

# ランダムミニバッチ学習を行う
print("--- 訓練開始 ---")
for epoch in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)

    if epoch % 100 == 0:
        train_list = [accuracy, acc_summary_train, loss_summary_train]
        train_list.extend(w_hists)
        train_list.extend(b_hists)
        train_list.extend(a_hists)
        train_list.append(y_hist)

        # トレーニングセット
        result = session.run(train_list, feed_dict={x: batch_xs, label: batch_ys})
        for i in range(1, len(result)):
            writer.add_summary(result[i], epoch)
        print("Train accuracy at epoch %s: %s" % (epoch, result[0]))

        # バリデーションセット
        val_list = [accuracy, acc_summary_val, loss_summary_val]
        result = session.run(val_list, feed_dict={x: mnist.validation.images, label: mnist.validation.labels})
        for i in range(1, len(result)):
            writer.add_summary(result[i], epoch)
        print("Validation accuracy at epoch %s: %s" % (epoch, result[0]))

        # テストセット
        test_list = [accuracy, acc_summary_test, loss_summary_test]
        result = session.run(test_list, feed_dict={x: mnist.test.images, label: mnist.test.labels})
        for i in range(1, len(result)):
            writer.add_summary(result[i], epoch)

        print("Test accuracy at epoch %s: %s" % (epoch, result[0]))
    else:
        result = session.run(train_step, feed_dict={x: batch_xs, label: batch_ys})
print("--- 訓練完了 ---")

# 終了時刻
end_time = time.time()
print("かかった時間: " + str(end_time - start_time))
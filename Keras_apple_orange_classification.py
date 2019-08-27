# Kerasによる、ものすごくシンプルな画像分類（りんごとオレンジ）
# https://qiita.com/hiroeorz@github/items/ecb39ed4042ebdc0a957


from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adagrad
from keras.optimizers import Adam
import numpy as np
from PIL import Image
import os

# 学習用のデータを作る.
image_list = []
label_list = []

# ./data/train 以下のorange,appleディレクトリ以下の画像を読み込む。
for dir in os.listdir("data/train"):
    if dir == ".DS_Store":
        continue

    dir1 = "data/train/" + dir
    label = 0

    if dir == "one_apple":    # appleはラベル0
        label = 0
    elif dir == "many_apples": # orangeはラベル1
        label = 1

    for file in os.listdir(dir1):
        if file != ".DS_Store":
            # 配列label_listに正解ラベルを追加(りんご:0 オレンジ:1)
            label_list.append(label)
            filepath = dir1 + "/" + file
            # 画像を25x25pixelに変換し、1要素が[R,G,B]3要素を含む配列の25x25の２次元配列として読み込む。
            # [R,G,B]はそれぞれが0-255の配列。
            image = np.array(Image.open(filepath).resize((25, 25)))
            print(filepath)
            # 配列を変換し、[[Redの配列],[Greenの配列],[Blueの配列]] のような形にする。
            # 默认是轴按照0，1，2输出
            # (225, 225, 3)
            # 改成image.transpose(2, 0, 1)后 数组维度改变
            # (3, 225, 225)
            image = image.transpose(2, 0, 1)
            # さらにフラットな1次元配列に変換。最初の1/3はRed、次がGreenの、最後がBlueの要素がフラットに並ぶ。
            # 这个是各个维度的数值提取出来相乘
            print(image.shape)
            # (4, 25, 25)
            # 这里的[0]表示的是数组的第1个数组，取出来。
            image = image.reshape(1, image.shape[0] * image.shape[1] * image.shape[2]).astype("float32")[0]
            # 出来上がった配列をimage_listに追加。
            # （RGB的数值在0-255区间）第一个数组所有的值都除以255将数值变成0到1之间的数，便于运算
            image_list.append(image / 255.)

# kerasに渡すためにnumpy配列に変換。　type:<class 'numpy.ndarray'>
image_list = np.array(image_list)

# ラベルの配列を1と0からなるラベル配列に変更
# 0 -> [1,0], 1 -> [0,1] という感じ。
Y = to_categorical(label_list)

# モデルを生成してニューラルネットを構築
model = Sequential()
model.add(Dense(200, input_dim=1875))
model.add(Activation("relu"))
model.add(Dropout(0.2))

model.add(Dense(200))
model.add(Activation("relu"))
model.add(Dropout(0.2))

model.add(Dense(2))
model.add(Activation("softmax"))

# オプティマイザにAdamを使用
opt = Adam(lr=0.001)
# モデルをコンパイル
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
# 学習を実行。10%はテストに使用。
model.fit(image_list, Y, nb_epoch=1500, batch_size=100, validation_split=0.1)

# テスト用ディレクトリ(./data/train/)の画像でチェック。正解率を表示する。
total = 0.
ok_count = 0.

for dir in os.listdir("data/train"):
    if dir == ".DS_Store":
        continue

    dir1 = "data/test/" + dir
    label = 0

    if dir == "one_apple":
        label = 0
    elif dir == "many_apples":
        label = 1

    for file in os.listdir(dir1):
        if file != ".DS_Store":
            label_list.append(label)
            filepath = dir1 + "/" + file
            image = np.array(Image.open(filepath).resize((25, 25)))
            print(filepath)
            image = image.transpose(2, 0, 1)
            image = image.reshape(1, image.shape[0] * image.shape[1] * image.shape[2]).astype("float32")[0]
            result = model.predict_classes(np.array([image / 255.]))
            print(result)
            print("标注值:", label, "机器预测结果:", result[0])

            total += 1.

            if label == result[0]:
                ok_count += 1.

print("机器预测结果和标注值匹配度: ", ok_count / total * 100, "%")
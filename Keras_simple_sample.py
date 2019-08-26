from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.utils.np_utils import to_categorical
import numpy as np

# 学習のためのデータ。
# 今回、Xは[0,0]または[1,1]の2種類。
# Yは0または1の2種類
# X:[0,0] => Y:0
# X:[1,1] => Y:1
# という対応になっている
X_list = [[0, 0], [1, 1], [0, 0], [1, 1], [1, 1], [1, 1]]
Y_list = [0, 1, 0, 1, 1, 1]

# kerasのmodelに渡す前にXをnumpyのarrayに変換する。
X = np.array(X_list)

# Yの各要素を0と1からなるリストに変換する。
# 0は0番目の要素のみ1で他は0の配列: [1,0]
# 1は1番目の要素のみ1で他は0の配列: [0,1]
# に変換される。
# すなわち
# [0, 1, 0, 1, 1, 1] => [[1,0], [0,1], [1,0], [0,1], [0,1], [0,1]]
# に変換される。
Y = to_categorical(Y_list)

# 学習のためのモデルを作る
model = Sequential()
# 全結合層(2層->10層)
model.add(Dense(input_dim=2, output_dim=10))
# 活性化関数(ReLu関数)
model.add(Activation("relu"))
# 全結合層(10層->2層)
model.add(Dense(output_dim=2))
# 活性化関数(softmax関数)
model.add(Activation("softmax"))
# モデルをコンパイル
model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
# 学習を実行
model.fit(X, Y, nb_epoch=3000, batch_size=32)

# 学習したモデルで予測する。
# [1,1]=> 1([0,1]) 1番目のビットが立っている
# [0,0]=> 0([1,0]) 0番目のビットが立っている
# という予測になるはず...
results = model.predict_proba(np.array([[1, 1], [0, 0]]))
# 結果を表示
print("Predict:\n", results)

# Predict:
#  [[0.00369103 0.996309  ]
#  [0.9354615  0.0645385 ]]

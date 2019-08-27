# Deep Learning初心者がKerasを使って気象衛星画像から天気予報をやってみた
# https://qiita.com/smatsui@github/items/e2e479164c31dcee7e32

import os
import numpy
import pandas
import pickle
from PIL import Image

train_data_src_dir = "学習データを保存したディレクトリの絶対パス"
train_label_csv = "CSV形式の学習データの正解ラベルの絶対パス"
test_data_src_dir = "テストデータを保存したディレクトリの絶対パス"
test_label_csv = "CSV形式のテストデータの正解ラベルの絶対パス"

def img2nparray(file):
    img = Image.open(file, "r")
    array = numpy.asarray(img, dtype="uint8")
    array = array.reshape(array.shape[0], array.shape[1], 1)
    return array

def get_label_set(file):
    labels = pandas.read_csv(file, encoding="shift-jis")
    labels = labels[labels["東京の天気"].notnull()]

    return labels["東京の天気"].as_matrix()


def generate_dataset():
    print("Generating train data set")
    master_dataset = []
    files = os.listdir(train_data_src_dir)
    for file in files:
        master_dataset.append(img2nparray(train_data_src_dir + file))

    master_dataset = numpy.array(master_dataset)
    train_label_set = get_label_set(train_label_csv)
    train_set = master_dataset, train_label_set

    print("Generating test data set")
    test_dataset = []
    files = os.listdir(test_data_src_dir)
    for file in files:
        test_dataset.append(img2nparray(test_data_src_dir + file))

    test_dataset = numpy.array(test_dataset)
    test_label_set = get_label_set(test_label_csv)
    test_set = test_dataset, test_label_set

    return (master_dataset, train_label_set), (test_dataset, test_label_set)

if __name__ == '__main__':
    dataset = generate_dataset()

    print("Creating pickle file")
    f = open(os.path.dirname(train_label_csv) + os.sep + 'data.pkl.gz', 'wb')
    binary = pickle.dump(dataset, f, protocol=2)
    f.close()
    print("Created")
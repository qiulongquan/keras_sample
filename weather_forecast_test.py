# Deep Learning初心者がKerasを使って気象衛星画像から天気予報をやってみた
# https://qiita.com/smatsui@github/items/e2e479164c31dcee7e32

# 天気予報用の衛星画像
# http://himawari8.nict.go.jp/ja/himawari8-image.htm

# 天気予報衛星画像ダウンロード
# https://seg-web.nict.go.jp/wsdb_osndisk/shareDirDownload/bDw2maKV
#
# 気象庁CSVデータがダウンロード
# http://www.data.jma.go.jp/risk/obsdl/index.php#!table

import os
import numpy
import pandas
import pickle
from PIL import Image

test_data_src_dir = "/Users/t-lqiu/keras_sample/data/weather_forecast/test/image/"
test_label_csv = "/Users/t-lqiu/keras_sample/data/weather_forecast/test/csv/weather.csv"
train_data_src_dir = "/Users/t-lqiu/keras_sample/data/weather_forecast/train/image/"
train_label_csv = "/Users/t-lqiu/keras_sample/data/weather_forecast/train/csv/weather.csv"

def img2nparray(file):
    img = Image.open(file, "r")
    array = numpy.asarray(img, dtype="uint8")
    array = array.reshape(array.shape[0], array.shape[1], 3)
    return array

def get_label_set(file):
    labels = pandas.read_csv(file, encoding="utf-8")
    labels = labels[labels["東京の天気"].notnull()]

    return labels["東京の天気"].as_matrix()


def generate_dataset():
    print("Generating train data set")
    master_dataset = []
    files = os.listdir(train_data_src_dir)
    for file in files:
        print(train_data_src_dir + file)
        if file == '.DS_Store':
            continue
        master_dataset.append(img2nparray(train_data_src_dir + file))

    master_dataset = numpy.array(master_dataset)
    train_label_set = get_label_set(train_label_csv)
    train_set = master_dataset, train_label_set

    print("Generating test data set")
    test_dataset = []
    files = os.listdir(test_data_src_dir)
    for file in files:
        print(test_data_src_dir + file)
        if file == '.DS_Store':
            continue
        test_dataset.append(img2nparray(test_data_src_dir + file))

    test_dataset = numpy.array(test_dataset)
    test_label_set = get_label_set(test_label_csv)
    test_set = test_dataset, test_label_set

    return (master_dataset, train_label_set), (test_dataset, test_label_set)


if __name__ == '__main__':
    # print(os.path.dirname(train_label_csv))
    # 只显示目录的绝对地址不包括文件名
    # /Users/t-lqiu/keras_sample/data/weather_forecast/train/csv

    dataset = generate_dataset()

    print("Creating pickle file")
    f = open(os.path.dirname(train_label_csv) + os.sep + 'data.pkl.gz', 'wb')
    # pickle解释，pickle可以把object对象用文件的方式保存dump，然后用load装载object，并直接使用。
    # https://qiita.com/moroku0519/items/f1f4c059c28cb1575a93
    binary = pickle.dump(dataset, f, protocol=2)
    f.close()
    print("Created")
    print("文件生成路径:", os.path.dirname(train_label_csv) + os.sep + 'data.pkl.gz')

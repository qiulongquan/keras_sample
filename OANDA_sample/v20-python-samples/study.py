# yahoo finance
# https://finance.yahoo.com/quote/%5EN225/history?period1=1472655600&period2=1567263600&interval=1d&filter=history&frequency=1d

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from pandas.plotting import autocorrelation_plot
from pandas.plotting import scatter_matrix
# ダウンロードしてきたやつ
# INDEIES = ["N225",  # Nikkei 225, Japan
#            "HSI",   # Hang Seng, Hong Kong
#            "GDAXI",  # DAX, German
#            "DJI",   # Dow, US
#            "GSPC",  # S&P 500, US
#            "BVSP",  # BOVESPA, Brazil
#            "IXIC"  # IXIC
#          ]

INDEIES = [
           "GSPC",  # S&P 500, US
            "N225",
            "HSI",
            "BVSP",
            "IXIC"
          ]

def study():
    closing = pd.DataFrame()
    closing_bk = pd.DataFrame()
    for index in INDEIES:
        # na_valuesは文字列"null"のとき空として扱う CSVみるとnullって書いてあります。
        df = pd.read_csv("./data/" + index + ".csv", na_values=["null"])
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
        closing[index] = df["Close"]
        closing_bk[index] = df["Close"]
        print("original:", closing[index])
    #空の部分は古いので埋める。
    closing = closing.fillna(method="ffill")
    print(closing.describe())
    for index in INDEIES:
        closing[index] = closing[index] / max(closing[index])
        closing_bk[index] = closing_bk[index] / max(closing_bk[index])
        closing[index] = np.log(closing[index] / closing[index].shift())
    #グラフ表示
    closing.plot()
    # plt.show()

    #自己相関
    fig = plt.figure()
    # 通过设定width和height来设定输出图片的尺寸
    fig.set_figwidth(5)
    fig.set_figheight(5)
    for index in INDEIES:
        autocorrelation_plot(closing[index], label=index)
    # plt.show()

    # 自己相関
    fig = plt.figure()
    # 通过设定width和height来设定输出图片的尺寸
    fig.set_figwidth(5)
    fig.set_figheight(5)
    for index in INDEIES:
        x = closing_bk[index]
        print("x:", x)
        y = np.sin(x)
        print("y:", y)
        plt.plot(x, y)
    # plt.show()

    #散布図行列
    # 通过设定figsize来设定输出图片的尺寸
    scatter_matrix(closing, figsize=(20, 20), diagonal='kde')
    plt.show()


if __name__ == "__main__":
    study()

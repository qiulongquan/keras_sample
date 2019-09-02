import pandas as pd
import numpy as np
import datetime

# df = pd.DataFrame({
#     'Col1a': [10, 20, 15, 30, 45],
#     'Col2a': [13, 23, 18, 33, 48],
#     'Col3a': [17, 27, 22, 37, 52]
# })
dates = pd.date_range('20190901',periods=6)
# df = pd.DataFrame(np.arange(6).reshape(6,1),index=['a','b','c','d','e','f'],columns=['value'])
df = pd.DataFrame(np.arange(6).reshape(6,1),index=dates,columns=['value'])
print(df)
print("--------------")

# shift函数没有其效果，value没有移动。
# https://www.cnblogs.com/iamxyq/p/6283334.html
# DataFrame创建数据方法
# https://morvanzhou.github.io/tutorials/data-manipulation/np-pd/3-1-pd-intro/

# df.shift(periods=3,freq=datetime.timedelta(1))
df.shift(periods=1, freq=None, axis=1)
print(df)
# print("start--显示相关度--:\n", df.corr())

print("="*50)
df1 = pd.DataFrame(np.arange(16).reshape(4,4),columns=['AA','BB','CC','DD'],index =pd.date_range('2012-06-01','2012-06-04'))
print(df1)
df1.shift(freq=datetime.timedelta(1))
print(df1)
df1.shift(freq=datetime.timedelta(-2))
print(df1)

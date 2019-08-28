# 关于 numpy数组 数组维度查看 数组维度改变 数组颠倒的实例
import numpy as np

a = np.arange(20)
print(a)
a = np.arange(20).shape
print(a)
a = np.arange(20).reshape(4, 5)
print(a)
# 第一个数组[0]全体值提取。
a1 = np.arange(20).reshape(4, 5)[0]
print("a1:", a1)
# 很多运算都需要在0到1之间的数值进行，所以取得的第一个数组全体都除以20，变成0到1之间的数值。
print("a1/20:", a1/20.0)
b = np.arange(20).reshape(4, 5).transpose(1, 0)
print(b)

print("b shape", b.shape)
print("b shape[0]", b.shape[0])
print("b shape[1]", b.shape[1])

b1 = b.reshape(1, b.shape[0] * b.shape[1]).astype("float32")[0]
print("b1:", b1)

list = [1,2,3,4,5,6]
list_array = np.array(list)
print("list_array:{},type:{}".format(list_array,type(list_array)))

print(np.array(list).reshape(2,3,1))

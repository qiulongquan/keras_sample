# scipy.misc.imread()被弃用，应该用imageio.imwrite()来替代
# 有warining，提示我们imread和imsave在后来的版本将会被弃用，
# 叫我们使用imageio.imread和imageio.imwrite
import imageio
import numpy as np
import matplotlib.pyplot as plt
im2 = imageio.imread('1.jpg')
print(im2.dtype)
print(im2.size)
print(im2.shape)
im = np.float64(im2)
print(im.dtype)
im = np.int8(im2)
print(im.dtype)

# output:
# uint8
# 71368704
# (5632, 4224, 3)
# float64
# int8

plt.imshow(im2)
plt.show()
print(im2)
imageio.imwrite('imageio.jpg', im2)

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import img_as_float
from skimage.io import imread
from scipy.ndimage import minimum_filter, maximum_filter

# Load ảnh (dạng float từ 0 -> 1)
img = img_as_float(imread('coins.tif', as_gray=True))
img_salt = img_as_float(imread('coins_salt.jpg', as_gray=True))
img_pepper = img_as_float(imread('coins_pepper.jpg', as_gray=True))

# Hiển thị ảnh gốc và ảnh nhiễu
FS = 15
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image', fontsize=FS)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(img_salt, cmap='gray')
plt.title('Salt Noise Image', fontsize=FS)
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(img_pepper, cmap='gray')
plt.title('Pepper Noise Image', fontsize=FS)
plt.axis('off')
plt.tight_layout()
plt.savefig('Original_and_Noisy_Images.jpg')
plt.show()

# Áp dụng lọc tối đa và tối thiểu (4x4 window)
size = (4, 4)
img_fix1 = minimum_filter(img_salt, size=size)   # xử lý ảnh nhiễu muối
img_fix2 = maximum_filter(img_pepper, size=size) # xử lý ảnh nhiễu tiêu
img_fix3 = maximum_filter(img_salt, size=size)   # lọc sai: dùng max cho muối
img_fix4 = minimum_filter(img_pepper, size=size) # lọc sai: dùng min cho tiêu

# Hiển thị kết quả lọc
plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1)
plt.imshow(img_fix1, cmap='gray')
plt.title('Salt Denoise Image', fontsize=FS)
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(img_fix2, cmap='gray')
plt.title('Pepper Denoise Image', fontsize=FS)
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(img_fix3, cmap='gray')
plt.title('Salt Wrongly Denoise Image', fontsize=FS)
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(img_fix4, cmap='gray')
plt.title('Pepper Wrongly Denoise Image', fontsize=FS)
plt.axis('off')
plt.tight_layout()
plt.savefig('Denoise_Images.jpg')
plt.show()

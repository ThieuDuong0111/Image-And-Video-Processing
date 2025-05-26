import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgba2rgb, rgb2gray
from skimage.morphology import disk, dilation, erosion
from skimage.util import img_as_float

# Đọc ảnh và xử lý RGBA → RGB → Grayscale
Img_rgba = imread('cliparts.png')
Img_rgb = rgba2rgb(Img_rgba)
Img_gray = rgb2gray(Img_rgb)
Img = img_as_float(Img_gray)

# Dilation và Erosion
se = disk(5)
Img_dilated = dilation(Img, se)
Img_eroded = erosion(Img, se)

# Trích xuất biên
ext_edge = Img_dilated - Img
int_edge = Img - Img_eroded
edge = Img_dilated - Img_eroded

# Hiển thị kết quả
FS = 15
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.imshow(Img, cmap='gray')
plt.title('Original Image', fontsize=FS)
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(ext_edge, cmap='gray')
plt.title('Edge_1 = External', fontsize=FS)
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(int_edge, cmap='gray')
plt.title('Edge_2 = Internal', fontsize=FS)
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(edge, cmap='gray')
plt.title('Edge_3 = External + Internal', fontsize=FS)
plt.axis('off')

plt.tight_layout()
plt.savefig('Boundary Detection using Dilation and Erosion.jpg')
plt.show()
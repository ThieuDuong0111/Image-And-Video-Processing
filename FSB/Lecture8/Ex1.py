import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from scipy.ndimage import sobel
from scipy.ndimage import convolve
from skimage.color import rgba2rgb, rgb2gray

# Đọc và chuyển ảnh sang grayscale nếu cần
Img = imread('bike.png')

# Nếu ảnh RGBA (4 kênh), chuyển sang RGB trước
if Img.shape[-1] == 4:
    from skimage.color import rgba2rgb
    Img = rgba2rgb(Img)

# Nếu ảnh RGB, chuyển sang grayscale
if len(Img.shape) == 3:
    Img = rgb2gray(Img)

Img = Img.astype(np.float64)

# Bộ lọc Sobel ngang và dọc
sobel_h = np.array([[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]])

sobel_v = sobel_h.T

# Áp dụng Sobel theo chiều ngang
filtered_Img1 = convolve(Img, sobel_h, mode='reflect')
filtered_Img1 = np.abs(filtered_Img1)
filtered_Img1 /= np.max(filtered_Img1)

# Áp dụng Sobel theo chiều dọc
filtered_Img2 = convolve(Img, sobel_v, mode='reflect')
filtered_Img2 = np.abs(filtered_Img2)
filtered_Img2 /= np.max(filtered_Img2)

# Tổng hợp độ lớn gradient
filtered_Img3 = filtered_Img1**2 + filtered_Img2**2
log_filtered_Img3 = np.log(filtered_Img3 + 1)
log_filtered_Img3 /= np.max(log_filtered_Img3)

# Ngưỡng
bw_edge1 = log_filtered_Img3 > 0.01
bw_edge2 = log_filtered_Img3 > 0.02
bw_edge3 = log_filtered_Img3 > 0.04

# Hiển thị kết quả
FS = 15
plt.figure(figsize=(12, 6))

plt.subplot(2, 3, 1)
plt.imshow(Img, cmap='gray')
plt.title('Original Image', fontsize=FS)
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(filtered_Img1, cmap='gray')
plt.title('Sobel Horizontal', fontsize=FS)
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(filtered_Img2, cmap='gray')
plt.title('Sobel Vertical', fontsize=FS)
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(bw_edge1, cmap='gray')
plt.title('Mag > 0.01', fontsize=FS)
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(bw_edge2, cmap='gray')
plt.title('Mag > 0.02', fontsize=FS)
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(bw_edge3, cmap='gray')
plt.title('Mag > 0.04', fontsize=FS)
plt.axis('off')

plt.tight_layout()
plt.savefig('Sobel Operator.jpg')
plt.show()
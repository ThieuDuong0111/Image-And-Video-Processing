import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import imageio.v2 as imageio
import os

# Đọc ảnh và chuyển sang kiểu float
image_path = 'man_face.png'  # Đảm bảo file ảnh nằm đúng thư mục
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Không tìm thấy tệp ảnh: {image_path}")

Img = imageio.imread(image_path).astype(float)

# Nếu ảnh có 3 kênh (RGB), chuyển sang grayscale
if Img.ndim == 3:
    Img = Img.mean(axis=2)

# Tạo mặt nạ Prewitt
h = np.array([[1, 0, -1],
              [1, 0, -1],
              [1, 0, -1]])

# Lọc theo hướng ngang
filtered_Img1 = convolve(Img, h, mode='nearest')
filtered_Img1 = np.abs(filtered_Img1)
filtered_Img1 = filtered_Img1 / np.max(filtered_Img1)

# Lọc theo hướng dọc (transpose của h)
filtered_Img2 = convolve(Img, h.T, mode='nearest')
filtered_Img2 = np.abs(filtered_Img2)
filtered_Img2 = filtered_Img2 / np.max(filtered_Img2)

# Tổng hợp cả hai hướng
filtered_Img3 = filtered_Img1 ** 2 + filtered_Img2 ** 2
log_filtered_Img3 = np.log(filtered_Img3 + 1)
log_filtered_Img3 = log_filtered_Img3 / np.max(log_filtered_Img3)

# Ngưỡng biên
bw_edge1 = log_filtered_Img3 > 0.01
bw_edge2 = log_filtered_Img3 > 0.02
bw_edge3 = log_filtered_Img3 > 0.04

# Vẽ ảnh
FS = 15
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(Img.astype(np.uint8), cmap='gray')
plt.title('Original Image', fontsize=FS)

plt.subplot(2, 3, 2)
plt.imshow(filtered_Img1, cmap='gray')
plt.title('Prewitt Horizontal', fontsize=FS)

plt.subplot(2, 3, 3)
plt.imshow(filtered_Img2, cmap='gray')
plt.title('Prewitt Vertical', fontsize=FS)

plt.subplot(2, 3, 4)
plt.imshow(bw_edge1, cmap='gray')
plt.title('Mag > 0.01', fontsize=FS)

plt.subplot(2, 3, 5)
plt.imshow(bw_edge2, cmap='gray')
plt.title('Mag > 0.02', fontsize=FS)

plt.subplot(2, 3, 6)
plt.imshow(bw_edge3, cmap='gray')
plt.title('Mag > 0.04', fontsize=FS)

plt.tight_layout()
plt.savefig('Prewitt_Operator.jpg')
plt.show()
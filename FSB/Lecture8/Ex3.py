import numpy as np
import matplotlib.pyplot as plt
from skimage import io, feature, color
import os

# Đọc ảnh
image_path = 'bike.png'  # Đảm bảo ảnh nằm đúng thư mục
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")

Img = io.imread(image_path)

# Nếu ảnh là RGB, chuyển về grayscale
if Img.ndim == 3:
    Img_gray = color.rgb2gray(Img)
else:
    Img_gray = Img / 255.0  # Chuẩn hóa ảnh uint8 sang [0,1]

# Các giá trị sigma
sigma_values = [np.sqrt(2), 2 * np.sqrt(2), 4 * np.sqrt(2)]
threshold = 0.2  # threshold thấp cho edge detection (low hysteresis threshold)

# Vẽ ảnh
FS = 15
plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1)
plt.imshow(Img, cmap='gray')
plt.title('Original Image', fontsize=FS)
plt.axis('off')

# Canny edge detection với từng sigma
for i, sigma in enumerate(sigma_values):
    edges = feature.canny(Img_gray, sigma=sigma, low_threshold=threshold, high_threshold=2 * threshold)
    plt.subplot(2, 2, i + 2)
    plt.imshow(edges, cmap='gray')
    plt.title(f'sigma = {sigma:.2f}', fontsize=FS)
    plt.axis('off')

plt.tight_layout()
plt.savefig('Canny_Edge_Detector.jpg')
plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

# Load ảnh grayscale
img = cv2.imread('coins.tif', cv2.IMREAD_GRAYSCALE)

# Thêm nhiễu Gaussian với mean=0, var=0.05
mean = 0
var = 0.05
sigma = var**0.5
gaussian_noise = np.random.normal(mean, sigma, img.shape)
img_noise = img.astype(np.float32) / 255.0 + gaussian_noise
img_noise = np.clip(img_noise, 0, 1) * 255
img_noise = img_noise.astype(np.uint8)

# Hiển thị ảnh gốc và ảnh nhiễu
FS = 15
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image', fontsize=FS)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_noise, cmap='gray')
plt.title('Noisy Image', fontsize=FS)
plt.axis('off')
plt.tight_layout()
plt.savefig('Original_vs_Noisy_Image.jpg')
plt.show()

# Tạo kernel trung bình (mean filter)
f1 = np.ones((3, 3), dtype=np.float32) / 9.0
f2 = np.ones((5, 5), dtype=np.float32) / 25.0

# Áp dụng lọc trung bình với 2 kernel khác nhau
img_denoise1 = convolve(img_noise, f1, mode='reflect')
img_denoise2 = convolve(img_noise, f2, mode='reflect')

# Hiển thị ảnh đã lọc
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(img_denoise1, cmap='gray')
plt.title('Denoise with 3x3 kernel', fontsize=FS)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_denoise2, cmap='gray')
plt.title('Denoise with 5x5 kernel', fontsize=FS)
plt.axis('off')
plt.tight_layout()
plt.savefig('Filtered_Image_with_Different_Kernel_Sizes.jpg')
plt.show()
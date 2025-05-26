import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_float
from skimage.io import imread
from skimage.util import random_noise
from scipy.signal import convolve2d, butter, filtfilt
from scipy.ndimage import convolve
import cv2

# Load ảnh và chuyển sang dạng float [0, 1]
Img = img_as_float(imread('eagle.jfif'))

# Tạo hiệu ứng blur (motion blur)
LEN = 3
THETA = 5
def motion_blur_psf(length, angle):
    size = max(3, length)
    psf = np.zeros((size, size))
    center = size // 2
    rad = np.deg2rad(angle)
    dx = np.cos(rad)
    dy = np.sin(rad)

    for i in range(length):
        x = int(center + (i - length // 2) * dx)
        y = int(center + (i - length // 2) * dy)
        if 0 <= y < size and 0 <= x < size:
            psf[y, x] = 1
    psf /= psf.sum()
    return psf

PSF = motion_blur_psf(LEN, THETA)

# Áp dụng blur (convolution)
def apply_filter_rgb(image, kernel, mode='reflect'):
    if image.ndim == 2:
        return convolve(image, kernel, mode=mode)
    else:
        result = np.zeros_like(image)
        for c in range(3):
            result[:, :, c] = convolve(image[:, :, c], kernel, mode=mode)
        return result

blurred_Img = apply_filter_rgb(Img, PSF, mode='reflect')

# Thêm nhiễu Gaussian
noise_mean = 0
noise_var = 0.05
noisy_blurred_Img = random_noise(blurred_Img, mode='gaussian', mean=noise_mean, var=noise_var)

# Hiển thị ảnh
FS = 15
plt.figure(1, figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(Img)
plt.title('Original Image', fontsize=FS)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(blurred_Img)
plt.title('Blurry Image', fontsize=FS)
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(noisy_blurred_Img)
plt.title('Noisy Blurry Image', fontsize=FS)
plt.axis('off')

plt.tight_layout()
plt.savefig('Noisy Blurry Image.jpg')
plt.show()

# Tính MSE trước khi phục hồi
mse1 = 255 * (noisy_blurred_Img - Img)
print(f"Mean Square Error before restoration: {np.sqrt(np.mean(mse1 ** 2)):.2f}")

# Áp dụng bộ lọc Butterworth
# Tạo filter Butterworth (1D), sau đó áp dụng cho từng hàng và cột
b, a = butter(N=5, Wn=0.4)

def butterworth_filter_image(img, b, a):
    if img.ndim == 2:
        # ảnh grayscale
        temp = filtfilt(b, a, img, axis=0)
        filtered = filtfilt(b, a, temp, axis=1)
        return filtered
    elif img.ndim == 3:
        filtered = np.zeros_like(img)
        for c in range(3):
            temp = filtfilt(b, a, img[:, :, c], axis=0)
            filtered[:, :, c] = filtfilt(b, a, temp, axis=1)
        return filtered
    else:
        raise ValueError("Unsupported image shape")

filtered_Img = butterworth_filter_image(noisy_blurred_Img, b, a)

# Hiển thị ảnh sau khi lọc
plt.figure(2, figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(noisy_blurred_Img)
plt.title('Noisy Blurry Image', fontsize=FS)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(np.clip(filtered_Img, 0, 1))
plt.title('Denoise Deblurring Image', fontsize=FS)
plt.axis('off')

plt.tight_layout()
plt.savefig('Denoise Deblurring Image using Butterworth.jpg')
plt.show()

# Tính MSE sau khi phục hồi
mse2 = 255 * (filtered_Img - Img)
print(f"Mean Square Error after restoration: {np.sqrt(np.mean(mse2 ** 2)):.2f}")

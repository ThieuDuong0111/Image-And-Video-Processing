# Ex5 - Bài 2: Áp dụng bộ lọc Sharpen và Butterworth

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh và chuyển sang RGB
img = cv2.imread('weather.png')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Bộ lọc sharpen 3x3
kernel_sharpen = np.array([[0, -1, 0],
                           [-1, 9, -1],
                           [0, -1, 0]], dtype=np.float32) / 4.0

img_sharpened1 = cv2.filter2D(img_rgb, -1, kernel_sharpen, borderType=cv2.BORDER_REFLECT)

# Hàm tạo bộ lọc Butterworth
def butterworth_lowpass_filter(shape, cutoff, order):
    P, Q = shape
    u = np.arange(P)
    v = np.arange(Q)
    u = u - P // 2
    v = v - Q // 2
    V, U = np.meshgrid(v, u)
    D = np.sqrt(U**2 + V**2)
    H = 1 / (1 + (D / (cutoff * P))**(2 * order))
    return H

# Áp dụng Butterworth cho từng kênh màu
def apply_butterworth_filter(img_rgb, cutoff=0.4, order=5):
    channels = cv2.split(img_rgb)
    filtered = []

    for c in channels:
        f = np.fft.fft2(c)
        fshift = np.fft.fftshift(f)
        H = butterworth_lowpass_filter(c.shape, cutoff, order)
        f_filtered = fshift * H
        img_back = np.abs(np.fft.ifft2(np.fft.ifftshift(f_filtered)))
        img_back = np.clip(img_back, 0, 255).astype(np.uint8)
        filtered.append(img_back)

    return cv2.merge(filtered)

img_sharpened2 = apply_butterworth_filter(img_rgb)

# Hiển thị kết quả
plt.figure(figsize=(10, 8))
plt.subplot(1, 2, 1)
plt.imshow(img_sharpened1)
plt.title('Sharpened Image 1 (Filter)')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_sharpened2)
plt.title('Sharpened Image 2 (Butterworth)')
plt.axis('off')

plt.tight_layout()
plt.savefig('ex5_2.jpeg')
plt.show()
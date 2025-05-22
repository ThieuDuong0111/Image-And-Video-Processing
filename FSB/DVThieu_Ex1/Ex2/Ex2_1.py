import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# Đọc ảnh xám
img = cv2.imread('apple.jpeg', cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Không tìm thấy file 'apple.jpg'.")

FS = 12  # font size

# a. Logarithmic Transformation
c = 256 / np.log(512)
log_mapping = c * np.log1p(np.arange(256))  # Ánh xạ log cho các mức xám từ 0-255
log_mapping = np.clip(log_mapping, 0, 255).astype(np.uint8)

log_transformed = log_mapping[img]

# b. Piecewise Linear Transformation
LUT = np.zeros(256, dtype=np.uint8)
for r in range(256):
    if r <= 93:
        LUT[r] = np.clip(2 * r + 10, 0, 255)
    elif r <= 168:
        LUT[r] = np.clip(r - 5, 0, 255)
    elif r <= 214:
        LUT[r] = r
    else:
        LUT[r] = 255

piecewise_img = LUT[img]

# ========== Hiển thị ==========
plt.figure(figsize=(8, 10))

# 1. Original Image
plt.subplot(3, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Original Image", fontsize=FS)
plt.axis('off')

# 2. Log Mapping Function
plt.subplot(3, 2, 3)
plt.plot(log_mapping, color='green')
plt.title("Log Mapping Function", fontsize=FS)

# 3. Adjusted Image using LMF
plt.subplot(3, 2, 4)
plt.imshow(log_transformed, cmap='gray')
plt.title("Adjusted Image using LMF", fontsize=FS)
plt.axis('off')

# 4. Piecewise Linear Mapping Function
plt.subplot(3, 2, 5)
plt.plot(LUT, color='blue')
plt.title("Piecewise Linear Mapping Function", fontsize=FS)

# 5. Adjusted Image using PLMF
plt.subplot(3, 2, 6)
plt.imshow(piecewise_img, cmap='gray')
plt.title("Adjusted Image using PLMF", fontsize=FS)
plt.axis('off')

# 6. Rỗng (để cân layout)
plt.subplot(3, 2, 6)
plt.axis('off')

plt.tight_layout()
plt.savefig('ex2_1.jpg') # Lưu ảnh kết quả
plt.show()
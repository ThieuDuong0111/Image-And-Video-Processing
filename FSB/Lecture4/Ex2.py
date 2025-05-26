import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure

# Đọc ảnh xám
img = cv2.imread('moon.jpg', cv2.IMREAD_GRAYSCALE)
FS = 15  # font size

# === Trước khi cân bằng histogram ===
count, bins = np.histogram(img.flatten(), bins=256, range=[0, 256])
bins = bins[:-1]  # bỏ bin cuối

plt.figure(1, figsize=(12, 5))
plt.clf()
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image', fontsize=FS)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.bar(bins, count, width=1.0, edgecolor='black')
plt.grid(True)
plt.xlim([0, 255])
plt.ylim([0, max(count) + 500])
plt.xlabel('Gray Levels', fontsize=FS)
plt.ylabel('# of pixels', fontsize=FS)
plt.title('Histogram before Equalization', fontsize=FS)
plt.xticks(fontsize=FS)
plt.yticks(fontsize=FS)
plt.tight_layout()
plt.savefig('Before Equalization.jpg')

# === Sau khi cân bằng histogram ===
eq_img = cv2.equalizeHist(img)
count_eq, bins_eq = np.histogram(eq_img.flatten(), bins=256, range=[0, 256])
bins_eq = bins_eq[:-1]

plt.figure(2, figsize=(12, 5))
plt.clf()
plt.subplot(1, 2, 1)
plt.imshow(eq_img, cmap='gray')
plt.title('Equalized Image', fontsize=FS)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.bar(bins_eq, count_eq, width=1.0, edgecolor='black')
plt.grid(True)
plt.xlim([0, 255])
plt.ylim([0, max(count_eq) + 500])
plt.xlabel('Gray Levels', fontsize=FS)
plt.ylabel('# of pixels', fontsize=FS)
plt.title('Histogram after Equalization', fontsize=FS)
plt.xticks(fontsize=FS)
plt.yticks(fontsize=FS)
plt.tight_layout()
plt.savefig('After Equalization.jpg')

plt.show()
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh xám
img = cv2.imread('bay.jpg', cv2.IMREAD_GRAYSCALE)

# Tính histogram
count, bins = np.histogram(img.flatten(), bins=256, range=[0, 256])
bins = bins[:-1]  # bỏ bin cuối cùng để khớp chiều dài

FS = 15  # font size

# Vẽ ảnh và histogram
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image', fontsize=FS)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.bar(bins, count, width=1.0, edgecolor='black')
plt.grid(True)
plt.xlim([0, 255])
plt.ylim([0, max(count) + 500])
plt.xlabel('Gray Level', fontsize=FS)
plt.ylabel('# of pixels', fontsize=FS)
plt.xticks(fontsize=FS)
plt.yticks(fontsize=FS)

# Lưu hình
plt.tight_layout()
plt.savefig('Histogram.png')
plt.show()

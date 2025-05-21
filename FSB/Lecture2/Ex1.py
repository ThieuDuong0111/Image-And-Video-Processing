import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh màu
img = cv2.imread('nature.jpg')                      # OpenCV đọc ảnh theo BGR
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)      # Chuyển sang RGB để hiển thị bằng matplotlib
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # Chuyển sang ảnh xám

# Chuyển ảnh xám sang nhị phân (binary image) với ngưỡng 0.5
_, img_bw = cv2.threshold(img_gray, 127, 1, cv2.THRESH_BINARY)
img_bw = img_bw.astype(float)                       # Đổi về dạng float để giống im2double

# Hiển thị ảnh
fs = 15
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(img_rgb)
plt.title('Original Image', fontsize=fs)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(img_gray, cmap='gray')
plt.title('Grayscale Image', fontsize=fs)
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(img_bw, cmap='gray')
plt.title('Binary Image', fontsize=fs)
plt.axis('off')

plt.tight_layout()
plt.savefig('Image in Different Types.jpg')
plt.show()

# Lưu ảnh với chất lượng khác nhau
cv2.imwrite('nature100.jpg', img)                          # Mặc định (100%)
cv2.imwrite('nature75.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 75])  # 75%
cv2.imwrite('nature10.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 10])  # 10%

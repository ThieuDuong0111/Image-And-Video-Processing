import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh và chuyển thành float [0,1]
img = cv2.imread('bike.png', cv2.IMREAD_GRAYSCALE)
img = img.astype(np.float64) / 255.0

# Tạo các kernel lọc
h1 = np.ones((1, 10), dtype=np.float64) / 10      # Horizontal filter
h2 = np.ones((10, 1), dtype=np.float64) / 10      # Vertical filter
h3 = np.ones((10, 10), dtype=np.float64) / 100    # Box/window filter

# Thực hiện lọc ảnh với các border khác nhau
filtered_img1 = cv2.filter2D(img, -1, h1, borderType=cv2.BORDER_REFLECT)    # 'symmetric'
filtered_img2 = cv2.filter2D(img, -1, h2, borderType=cv2.BORDER_REPLICATE)  # 'replicate'
filtered_img3 = cv2.filter2D(img, -1, h3, borderType=cv2.BORDER_WRAP)       # 'circular'

# Hiển thị và lưu ảnh
FS = 15

# Horizontal Filtering
plt.figure(1)
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image', fontsize=FS)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(filtered_img1, cmap='gray')
plt.title('Horizontal Filtering', fontsize=FS)
plt.axis('off')

plt.tight_layout()
plt.savefig('Horizontal_Filtering.png')

# Vertical Filtering
plt.figure(2)
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image', fontsize=FS)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(filtered_img2, cmap='gray')
plt.title('Vertical Filtering', fontsize=FS)
plt.axis('off')

plt.tight_layout()
plt.savefig('Vertical_Filtering.png')

# Box Filtering
plt.figure(3)
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image', fontsize=FS)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(filtered_img3, cmap='gray')
plt.title('Box Filtering', fontsize=FS)
plt.axis('off')

plt.tight_layout()
plt.savefig('Box_Filtering.png')

plt.show()
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure

# Đọc ảnh và chuyển sang ảnh xám
img = cv2.imread('newspaper.jpg', cv2.IMREAD_GRAYSCALE)

# Global Histogram Equalization
eq_img = cv2.equalizeHist(img)

# Local Histogram Equalization (chia khối 98x294 và áp dụng histeq cho mỗi khối)
def local_hist_equalization(image, block_size=(98, 294)):
    h, w = image.shape
    local_eq_img = np.zeros_like(image)

    for i in range(0, h, block_size[0]):
        for j in range(0, w, block_size[1]):
            # Lấy block con
            block = image[i:min(i+block_size[0], h), j:min(j+block_size[1], w)]

            # Histogram equalization trên block
            block_eq = cv2.equalizeHist(block)

            # Chèn block vào ảnh kết quả
            local_eq_img[i:min(i+block_size[0], h), j:min(j+block_size[1], w)] = block_eq
    return local_eq_img

lc_img = local_hist_equalization(img, block_size=(98, 294))

# Hiển thị kết quả
FS = 15
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image', fontsize=FS)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(eq_img, cmap='gray')
plt.title('Global Equal.', fontsize=FS)
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(lc_img, cmap='gray')
plt.title('Local Equal.', fontsize=FS)
plt.axis('off')

plt.tight_layout()
plt.savefig('LocalHistogramEqualization.png')
plt.show()
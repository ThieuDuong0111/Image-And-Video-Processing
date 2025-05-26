import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import erosion, diamond

# Load ảnh và chuyển sang grayscale
Img = cv2.imread('fence.jpg')
Img_rgb = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
Img_gray = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)

# Threshold tự động (Otsu)
_, BW = cv2.threshold(Img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
BW_bool = BW > 0

# a. Erosion với SE hình kim cương, radius = 35
BW1 = erosion(BW_bool, diamond(35))

# b. Erosion với SE dạng chữ thập (cross), dài 101 (tức 50 mỗi bên)
length = 101
cross_se = np.zeros((length, length), dtype=np.uint8)
cross_se[length // 2, :] = 1  # hàng giữa
cross_se[:, length // 2] = 1  # cột giữa
BW2 = erosion(BW_bool, cross_se)

# Vẽ structuring element chữ thập
plt.figure(1)
plt.imshow(cross_se, cmap='gray')
plt.title('50-pixel Cross SE', fontsize=15)
plt.axis('off')
plt.savefig('Cross Structuring Element.jpg')

# Vẽ các kết quả xử lý
plt.figure(2, figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.imshow(Img_rgb)
plt.title('Original Image', fontsize=15)
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(BW_bool, cmap='gray')
plt.title('Binary Image', fontsize=15)
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(BW1, cmap='gray')
plt.title('Erosion w. Diamond', fontsize=15)
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(BW2, cmap='gray')
plt.title('Erosion w. Cross', fontsize=15)
plt.axis('off')

plt.tight_layout()
plt.savefig('Hole Detection with Erosion.jpg')
plt.show()
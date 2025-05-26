import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import erosion, disk, square, rectangle, diamond

def create_line_se(length, angle_deg):
    """Tạo structuring element dạng đường thẳng nghiêng"""
    se = np.zeros((length, length), dtype=np.uint8)
    if angle_deg == 0:
        se[length // 2, :] = 1
    elif angle_deg == 90:
        se[:, length // 2] = 1
    elif angle_deg == 45:
        np.fill_diagonal(se, 1)
    elif angle_deg == 135:
        np.fill_diagonal(np.fliplr(se), 1)
    else:
        raise ValueError("Chỉ hỗ trợ góc 0, 45, 90, hoặc 135 độ.")
    return se

# Load ảnh nhị phân
Img = cv2.imread('circles.png', cv2.IMREAD_GRAYSCALE)
_, Img_bin = cv2.threshold(Img, 127, 255, cv2.THRESH_BINARY)
Img_bool = Img_bin > 0  # Chuyển sang boolean để dùng với skimage

FS = 15

# a. Disk with radius = 50
BW1 = erosion(Img_bool, disk(50))

# b. Square with side = 60
BW2 = erosion(Img_bool, square(60))

# c. Rectangle with dimensions = [38, 48]
BW3 = erosion(Img_bool, rectangle(38, 48))

# d. Diamond with radius = 50
BW4 = erosion(Img_bool, diamond(50))

# e. Line with length = 30, angle = 45
line_se = create_line_se(30, 45)
BW5 = erosion(Img_bool, line_se)

# Hiển thị hình ảnh
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.ravel()
titles = [
    'Original Image', 'Disk', 'Square',
    'Rectangle', 'Diamond', 'Line'
]
images = [Img_bool, BW1, BW2, BW3, BW4, BW5]

for i in range(6):
    axes[i].imshow(images[i], cmap='gray')
    axes[i].set_title(titles[i], fontsize=FS)
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('Circle Counting using Erosion.jpg')
plt.show()
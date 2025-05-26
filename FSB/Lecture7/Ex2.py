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
        raise ValueError("Chỉ hỗ trợ các góc: 0, 45, 90, 135")
    return se

# Load ảnh nhị phân
Img = cv2.imread('binary_objects.jpg', cv2.IMREAD_GRAYSCALE)
_, Img_bin = cv2.threshold(Img, 127, 255, cv2.THRESH_BINARY)
Img_bool = Img_bin > 0  # Chuyển sang boolean để dùng với skimage

FS = 15

# a. Small disk with radius = 5
BW1 = erosion(Img_bool, disk(5))

# b. Large disk with radius = 15
BW2 = erosion(Img_bool, disk(15))

# c. Square with side = 7
BW3 = erosion(Img_bool, square(7))

# d. Rectangle with dimensions = [5, 10]
BW4 = erosion(Img_bool, rectangle(5, 10))

# e. Diamond with radius = 4
BW5 = erosion(Img_bool, diamond(4))

# f. Line with length = 10 and angle = 45 degrees
line_se = create_line_se(10, 45)
BW6 = erosion(Img_bool, line_se)

# Hiển thị kết quả
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.ravel()
titles = [
    'Small Disk', 'Large Disk', 'Square',
    'Rectangle', 'Diamond', 'Line'
]
results = [BW1, BW2, BW3, BW4, BW5, BW6]

for i in range(6):
    axes[i].imshow(results[i], cmap='gray')
    axes[i].set_title(titles[i], fontsize=FS)
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('Erosion with Different SEs.jpg')
plt.show()

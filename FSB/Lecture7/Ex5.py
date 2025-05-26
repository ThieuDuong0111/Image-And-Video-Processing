import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import square, dilation, erosion

# Đọc ảnh (dạng grayscale)
Img_gray = cv2.imread('man_face.png', cv2.IMREAD_GRAYSCALE)

# Binarize ảnh bằng ngưỡng thủ công: BW = Img < 112
BW = Img_gray < 112  # kiểu bool

# Structuring element: square(22)
se = square(22)

# Closing = Dilation rồi Erosion
Img_dilated = dilation(BW, se)
Img_closed = erosion(Img_dilated, se)

# Phần thay đổi do closing (xem các lỗ đã được lấp)
Img_diff = Img_closed.astype(np.uint8) - BW.astype(np.uint8)

# Hiển thị kết quả
FS = 15  # font size
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.imshow(BW, cmap='gray')
plt.title('Binary Image', fontsize=FS)
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(Img_dilated, cmap='gray')
plt.title('Dilated Image', fontsize=FS)
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(Img_closed, cmap='gray')
plt.title('Closed Image', fontsize=FS)
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(Img_diff, cmap='gray')
plt.title('Closing - Binary', fontsize=FS)
plt.axis('off')

plt.tight_layout()
plt.savefig('Hole Removal using Closing.jpg')
plt.show()
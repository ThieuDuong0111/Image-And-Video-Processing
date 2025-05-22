import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh màu (theo định dạng BGR của OpenCV)
img_bgr = cv2.imread('sea.jpg')

# Chuyển sang định dạng RGB (phù hợp để hiển thị bằng matplotlib)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Tách các kênh màu
R = img_rgb[:, :, 0]
G = img_rgb[:, :, 1]
B = img_rgb[:, :, 2]

# Tạo ảnh mới với thứ tự BRG
img_brg = np.stack((B, R, G), axis=2)

# Hiển thị ảnh gốc và ảnh BRG
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_brg)
plt.title('Reordered Image')
plt.axis('off')

plt.tight_layout()
plt.savefig('ex2_sea_brg.jpg')  # Lưu ảnh kết quả
plt.show()

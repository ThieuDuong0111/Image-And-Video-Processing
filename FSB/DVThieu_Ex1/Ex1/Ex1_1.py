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

# Tạo ảnh chỉ hiển thị kênh màu tương ứng
zero_channel = np.zeros_like(R)

img_red = np.stack((R, zero_channel, zero_channel), axis=2)
img_green = np.stack((zero_channel, G, zero_channel), axis=2)
img_blue = np.stack((zero_channel, zero_channel, B), axis=2)

# Hiển thị ảnh
plt.figure(figsize=(10, 8))

plt.subplot(2, 3, 2)
plt.imshow(img_rgb)
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(img_red)
plt.title('Red Component')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(img_green)
plt.title('Green Component')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(img_blue)
plt.title('Blue Component')
plt.axis('off')

plt.tight_layout()
plt.savefig('ex1_sea_tricolor_channels.jpg') # Lưu ảnh kết quả
plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh và chuyển sang định dạng float64 trong khoảng [0, 1]
img = cv2.imread('rose.jpg')        # Mặc định là ảnh BGR

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Chuyển sang RGB để hiển thị đúng màu
img = img.astype(np.float64) / 255.0        # Chuẩn hóa về [0, 1]

# Điều chỉnh độ sáng
scale = 0.75  # <1 là tối hơn, >1 là sáng hơn
scaled_img = img * scale
scaled_img = np.clip(scaled_img, 0, 1)  # Giới hạn về [0, 1]

# Hiển thị ảnh gốc và ảnh đã điều chỉnh
FS = 15
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('Original Image', fontsize=FS)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(scaled_img)
plt.title('Darker Image', fontsize=FS)
plt.axis('off')

# Lưu ảnh kết quả (chuyển về uint8 trước)
scaled_img_uint8 = (scaled_img * 255).astype(np.uint8)
cv2.imwrite('Darker.jpg', cv2.cvtColor(scaled_img_uint8, cv2.COLOR_RGB2BGR))

plt.tight_layout()
plt.show()

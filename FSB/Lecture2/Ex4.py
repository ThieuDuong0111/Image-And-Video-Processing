import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh và chuyển về định dạng float64 [0, 1]
img = cv2.imread('waterfall.jpg')  # Ảnh được đọc ở định dạng BGR

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Chuyển sang RGB
img = img.astype(np.float64) / 255.0        # Chuẩn hóa ảnh về [0, 1]

# Áp dụng biến đổi gamma
gamma = 0.2  # gamma < 1 làm ảnh sáng hơn và giảm độ tương phản
enhanced_img = np.power(img, gamma)

# Hiển thị ảnh gốc và ảnh sau khi xử lý
FS = 15
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('Original Image', fontsize=FS)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(enhanced_img)
plt.title('Decrease Contrast', fontsize=FS)
plt.axis('off')

# Lưu ảnh kết quả
enhanced_img_uint8 = (enhanced_img * 255).astype(np.uint8)
cv2.imwrite('Decrease.png', cv2.cvtColor(enhanced_img_uint8, cv2.COLOR_RGB2BGR))

plt.tight_layout()
plt.show()
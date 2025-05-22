import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh
img_bgr = cv2.imread('sea.jpg')

# Chuyển sang RGB để hiển thị với matplotlib
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Áp dụng gamma correction (gamma = 1.2)
gamma = 1.2
gamma_corrected = 255.0 * (img_rgb / 255.0) ** gamma
gamma_corrected = np.clip(gamma_corrected, 0, 255).astype(np.uint8)

# Làm tối ảnh đi 80% (chỉ giữ lại 20% độ sáng)
darker_img = (gamma_corrected * 0.2).astype(np.uint8)

# Hiển thị 2 hình
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(darker_img)
plt.title("Enhanced Image")
plt.axis('off')

plt.tight_layout()
plt.savefig('ex3_sea_enhanced.jpg') # Lưu ảnh kết quả
plt.show()

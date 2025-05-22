import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc 2 ảnh màu
img1 = cv2.imread('cat_a.png')
img2 = cv2.imread('cat_b.png')

if img1 is None or img2 is None:
    raise FileNotFoundError("Không tìm thấy file 'cat_a.png' hoặc 'cat_b.png'.")

# Resize ảnh về cùng kích thước nếu khác nhau
if img1.shape != img2.shape:
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

# Tính hiệu tuyệt đối giữa 2 ảnh
diff_color = cv2.absdiff(img1, img2)

# Tạo ảnh xám
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
diff_gray = cv2.absdiff(gray1, gray2)

# Tăng tương phản 40% và tăng sáng 200%
enhanced_gray = cv2.convertScaleAbs(diff_gray, alpha=1.4, beta=200)

# Chuyển BGR → RGB để hiển thị đúng màu trên matplotlib
diff_color_rgb = cv2.cvtColor(diff_color, cv2.COLOR_BGR2RGB)

# Vẽ 3 hình: Màu, Xám, Xám tăng tương phản
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(diff_color_rgb)
plt.title("Color Differences")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(diff_gray, cmap='gray')
plt.title("Grayscale Differences")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(enhanced_gray, cmap='gray')
plt.title("Enhanced Grayscale")
plt.axis("off")

plt.tight_layout()
plt.savefig("ex3_2.jpg")  # Lưu hình tổng hợp
plt.show()
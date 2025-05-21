import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh gốc và ảnh có chữ
img_notext = cv2.imread('gradient.jpg', cv2.IMREAD_GRAYSCALE)  # ảnh gốc
img_withtext = cv2.imread('gradient_with_text.jpg', cv2.IMREAD_GRAYSCALE)  # ảnh có chữ

# Resize nếu kích thước không khớp
if img_notext.shape != img_withtext.shape:
    img_withtext = cv2.resize(img_withtext, (img_notext.shape[1], img_notext.shape[0]))

# Chuyển về float và thực hiện phép chia pixel-wise
withtext_float = img_withtext.astype(np.float32)
notext_float = img_notext.astype(np.float32)

# Tránh chia cho 0 bằng cách cộng epsilon nhỏ
epsilon = 1e-5
D = withtext_float / (notext_float + epsilon)

# Ngưỡng để xác định phần có chữ (vì khi có chữ thì pixel bị thay đổi => tỉ lệ > 1)
D_binary = D > 1.0

# Vẽ ảnh
FS = 15
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img_notext, cmap='gray')
plt.title("Without text", fontsize=FS)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(img_withtext, cmap='gray')
plt.title("With text", fontsize=FS)
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(D_binary, cmap='gray')
plt.title("Detected text", fontsize=FS)
plt.axis('off')

# Lưu ảnh kết quả
plt.tight_layout()
plt.savefig("Detected Text.jpg")
plt.show()

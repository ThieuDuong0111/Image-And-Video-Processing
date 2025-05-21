import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh và chuyển sang kiểu float trong khoảng [0,1]
I1 = cv2.imread('earth1.jpg')
I2 = cv2.imread('earth2.jpg')

# Kiểm tra nếu ảnh không đọc được
if I1 is None or I2 is None:
    raise FileNotFoundError("Không tìm thấy một trong hai ảnh. Kiểm tra lại tên file hoặc đường dẫn.")

# Resize I2 về cùng kích thước với I1 nếu khác kích thước
if I1.shape != I2.shape:
    I2 = cv2.resize(I2, (I1.shape[1], I1.shape[0]))

# Chuyển ảnh về kiểu float32 trong khoảng [0,1]
I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2RGB) / 255.0
I2 = cv2.cvtColor(I2, cv2.COLOR_BGR2RGB) / 255.0

# Nhân chồng các ảnh lại
I = I1 * I2 * I1 * I2  # tương đương với immultiply(immultiply(immultiply(I1,I2),I1),I2)

# Hiển thị kết quả
FS = 15
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(I1)
plt.title('2-D Image', fontsize=FS)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(I)
plt.title('3-D Effect Image', fontsize=FS)
plt.axis('off')

# Lưu kết quả
plt.tight_layout()
plt.savefig('3-D Effect.png')
plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc hai ảnh
Ia = cv2.imread('prarie.jpg')
Ib = cv2.imread('giraffe.jpg')

# Resize ảnh Ib về cùng kích thước với Ia
Ib_resized = cv2.resize(Ib, (Ia.shape[1], Ia.shape[0]))

# Cộng hai ảnh
Ic = cv2.add(Ia, Ib_resized)  # Hàm này tự động cắt ngưỡng giá trị pixel về 255 nếu vượt quá

# Chuyển sang RGB để hiển thị bằng matplotlib
Ia_rgb = cv2.cvtColor(Ia, cv2.COLOR_BGR2RGB)
Ib_rgb = cv2.cvtColor(Ib, cv2.COLOR_BGR2RGB)
Ic_rgb = cv2.cvtColor(Ic, cv2.COLOR_BGR2RGB)

# Hiển thị ảnh
FS = 15
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(Ia_rgb)
plt.title('prarie', fontsize=FS)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(Ib_rgb)
plt.title('giraffe', fontsize=FS)
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(Ic_rgb)
plt.title('Combination Image', fontsize=FS)
plt.axis('off')

# Lưu kết quả
cv2.imwrite('Addition Image.jpg', Ic)  # Lưu bằng định dạng BGR
plt.tight_layout()
plt.show()

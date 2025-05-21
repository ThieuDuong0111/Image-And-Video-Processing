import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh và chuyển về dạng float trong khoảng [0,1]
live = cv2.imread('live.jpg')
mask = cv2.imread('mask.jpg')

if live is None or mask is None:
    raise FileNotFoundError("Không tìm thấy ảnh 'live.jpg' hoặc 'mask.jpg'.")

# Resize mask về cùng kích thước nếu cần
if live.shape != mask.shape:
    mask = cv2.resize(mask, (live.shape[1], live.shape[0]))

# Chuyển sang RGB để hiển thị bằng matplotlib và scale [0,1]
live = cv2.cvtColor(live, cv2.COLOR_BGR2RGB) / 255.0
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB) / 255.0

# Trừ ảnh bằng absdiff rồi nâng mũ 0.4
diff_img = np.abs(live - mask) ** 0.4

# Vẽ ảnh
FS = 15
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(live)
plt.title("live", fontsize=FS)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(mask)
plt.title("mask", fontsize=FS)
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(diff_img)
plt.title("Subtraction Image", fontsize=FS)
plt.axis('off')

# Lưu kết quả
plt.tight_layout()
plt.savefig("Subtraction Image.jpg")
plt.show()

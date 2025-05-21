import cv2
import numpy as np
import matplotlib.pyplot as plt

FS = 15  # fontsize cho caption

# Load ảnh grayscale (mặc định imread màu)
Img = cv2.imread('lion.jpg', cv2.IMREAD_GRAYSCALE)
if Img is None:
    raise FileNotFoundError("Không tìm thấy file 'lion.jpg'.")

# ==== (b) Negative Transformation ====

# Vẽ đồ thị hàm biến đổi negative: mapping 0->255, 1->254, ..., 255->0
x = np.arange(255, -1, -1, dtype=np.uint8)
plt.figure(3)
plt.clf()
plt.plot(x, linewidth=1.5)
plt.xlim([0, 255])
plt.ylim([0, 255])
plt.grid(True)
plt.title("Negative Transformation Function")
plt.xlabel("Input Intensity")
plt.ylabel("Output Intensity")
plt.show()

# Áp dụng biến đổi negative: pixel mới = 255 - pixel cũ
Img2 = 255 - Img

# Hiển thị ảnh gốc và ảnh negative
plt.figure(4, figsize=(10, 5))
plt.clf()
plt.subplot(1, 2, 1)
plt.imshow(Img, cmap='gray', vmin=0, vmax=255)
plt.title('Original Image', fontsize=FS)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(Img2, cmap='gray', vmin=0, vmax=255)
plt.title('Negative', fontsize=FS)
plt.axis('off')

plt.tight_layout()
plt.savefig('Negative_Transformation.jpg')
plt.show()

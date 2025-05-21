import cv2
import numpy as np
import matplotlib.pyplot as plt

FS = 15  # fontsize caption

# Load ảnh grayscale
Img = cv2.imread('lion.jpg', cv2.IMREAD_GRAYSCALE)
if Img is None:
    raise FileNotFoundError("Không tìm thấy file 'lion.jpg'.")

# ====== Piece-wise Linear Transformation ======

# Tạo LUT theo đoạn piece-wise như trong Octave
LUT = np.zeros(256, dtype=np.uint8)

# LUT[0:101] = 2 * [0..100] + 10
LUT[0:101] = np.clip(2 * np.arange(101) + 10, 0, 255)

# LUT[101:201] = 175 (tương đương 102:201 trong Octave, index khác nhau vì Python bắt đầu từ 0)
LUT[101:201] = 175

# LUT[201:256] = 0.85 * [201..255] - 12
LUT[201:256] = np.clip(0.85 * np.arange(201, 256) - 12, 0, 255).astype(np.uint8)

# Vẽ đồ thị hàm biến đổi
plt.figure(1)
plt.clf()
plt.plot(LUT, linewidth=1.5)
plt.xlim([0, 255])
plt.ylim([0, 255])
plt.grid(True)
plt.title('Piece-wise Linear Transformation Function')
plt.xlabel('Input Intensity')
plt.ylabel('Output Intensity')
plt.show()

# Áp dụng LUT cho ảnh
Img4 = cv2.LUT(Img, LUT)

# Hiển thị ảnh gốc và ảnh biến đổi
plt.figure(2, figsize=(10, 5))
plt.clf()
plt.subplot(1, 2, 1)
plt.imshow(Img, cmap='gray', vmin=0, vmax=255)
plt.title('Original Image', fontsize=FS)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(Img4, cmap='gray', vmin=0, vmax=255)
plt.title('Piece-Wise Linear', fontsize=FS)
plt.axis('off')

plt.tight_layout()
plt.savefig('Piece_Wise_Linear_Transformation.jpg')
plt.show()

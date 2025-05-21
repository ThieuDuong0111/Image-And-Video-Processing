import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate

FS = 15  # Font size for titles

# ==== (a) Flipping an image up-down and left-right ====
img_flip = cv2.imread('atrium.jpg')
if img_flip is None:
    raise FileNotFoundError("Không tìm thấy file 'atrium.jpg'.")

# Chuyển sang RGB để hiển thị đúng màu
img_flip_rgb = cv2.cvtColor(img_flip, cv2.COLOR_BGR2RGB)

# Lật ảnh
img_ud = cv2.flip(img_flip_rgb, 0)  # Lật trên-dưới
img_lr = cv2.flip(img_flip_rgb, 1)  # Lật trái-phải

# Hiển thị kết quả
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1), plt.imshow(img_flip_rgb), plt.title('Original Image', fontsize=FS), plt.axis('off')
plt.subplot(1, 3, 2), plt.imshow(img_ud), plt.title('Flipped Up-Down Image', fontsize=FS), plt.axis('off')
plt.subplot(1, 3, 3), plt.imshow(img_lr), plt.title('Flipped Left-Right Image', fontsize=FS), plt.axis('off')
plt.tight_layout()
plt.savefig("Flipping Image.png")
plt.show()


# ==== (b) Rotating an image with an angle theta in degree ====
img_rotate = cv2.imread('eight.png')
if img_rotate is None:
    raise FileNotFoundError("Không tìm thấy file 'eight.png'.")

# Chuyển sang RGB nếu ảnh màu
if len(img_rotate.shape) == 3:
    img_rotate_rgb = cv2.cvtColor(img_rotate, cv2.COLOR_BGR2RGB)
else:
    img_rotate_rgb = img_rotate

theta = 90
# Sử dụng scipy để xoay ảnh, reshape=False để giữ nguyên kích thước nếu cần
img_rotated = rotate(img_rotate_rgb, angle=theta, reshape=True)

# Hiển thị ảnh xoay
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1), plt.imshow(img_rotate_rgb, cmap='gray'), plt.title('Original Image', fontsize=FS), plt.axis('off')
plt.subplot(1, 2, 2), plt.imshow(img_rotated, cmap='gray'), plt.title('Rotated Image', fontsize=FS), plt.axis('off')
plt.tight_layout()
plt.savefig("Rotating Image.png")
plt.show()

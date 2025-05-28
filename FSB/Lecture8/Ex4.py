import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, feature, transform
import os

# Đọc ảnh
image_path = 'line.png'  # Đảm bảo ảnh nằm trong cùng thư mục
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")

Img = io.imread(image_path)

# Nếu ảnh là ảnh màu, chuyển sang grayscale
if Img.ndim == 3:
    gray = color.rgb2gray(Img)
else:
    gray = Img / 255.0  # Nếu là ảnh đơn kênh uint8 thì chuẩn hóa

# Áp dụng Canny edge detector trước khi Hough Transform
edges = feature.canny(gray, sigma=2.0)

# Thực hiện Hough Transform
hspace, angles, dists = transform.hough_line(edges, theta=np.deg2rad(np.arange(-90, 91)))

# Hiển thị ảnh gốc
FS = 15
plt.figure(figsize=(6, 6))
plt.imshow(Img, cmap='gray')
plt.title("Original Image", fontsize=FS)
plt.axis('off')
plt.show()

# Hiển thị không gian Hough
plt.figure(figsize=(8, 6))
plt.imshow(hspace, extent=[np.rad2deg(angles[0]), np.rad2deg(angles[-1]), dists[-1], dists[0]],
           aspect='auto', cmap='hot')
plt.title('Hough Transform in 2-D', fontsize=FS)
plt.xlabel(r'$\theta$ (degrees)', fontsize=FS)
plt.ylabel(r'$\rho$', fontsize=FS)
plt.colorbar(label='Votes')
plt.grid(True)
plt.xticks(fontsize=FS - 2)
plt.yticks(fontsize=FS - 2)
plt.tight_layout()
plt.savefig('Hough_Transform_2D.png')
plt.show()
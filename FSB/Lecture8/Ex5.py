import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters, transform, feature, img_as_ubyte
from skimage.transform import hough_line, hough_line_peaks, rotate
import os

# Đọc ảnh gốc
image_path = 'paper.jpg'
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")

Img = io.imread(image_path)

# Chuyển sang grayscale nếu ảnh màu
if Img.ndim == 3:
    gray = color.rgb2gray(Img)
else:
    gray = Img / 255.0

# Nhị phân hóa ảnh
level = filters.threshold_otsu(gray)
bw_Img = gray < level  # giống 1 - im2bw

# Hiển thị ảnh gốc và ảnh nhị phân
FS = 15
plt.figure(1, figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(Img, cmap='gray')
plt.title("Original Image", fontsize=FS)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(bw_Img, cmap='gray')
plt.title("Binary Image", fontsize=FS)
plt.axis('off')
plt.tight_layout()
plt.savefig('Origin_vs_Binary_Images.png')
plt.show()

# Tính Hough Transform
hspace, angles, dists = hough_line(bw_Img, theta=np.deg2rad(np.arange(-90, 91)))

# Hiển thị Hough Transform
plt.figure(2, figsize=(8, 6))
plt.imshow(hspace, extent=[np.rad2deg(angles[0]), np.rad2deg(angles[-1]), dists[-1], dists[0]],
           aspect='auto', cmap='hot')
plt.title("Hough Transform Computation", fontsize=FS)
plt.xlabel(r'$\theta$ (degrees)', fontsize=FS)
plt.ylabel(r'$\rho$', fontsize=FS)
plt.colorbar(label='Votes')
plt.xticks(fontsize=FS - 2)
plt.yticks(fontsize=FS - 2)
plt.tight_layout()
plt.savefig('Hough_Transform_Computation.png')
plt.show()

# Phát hiện các peaks trong Hough Transform
peakNum = 8
h_peaks, a_peaks, d_peaks = hough_line_peaks(hspace, angles, dists, num_peaks=peakNum, threshold=0.2 * np.max(hspace))

# Vẽ các điểm peak
plt.figure(figsize=(8, 6))
plt.imshow(hspace, extent=[np.rad2deg(angles[0]), np.rad2deg(angles[-1]), dists[-1], dists[0]],
           aspect='auto', cmap='hot')
plt.plot(np.rad2deg(a_peaks), d_peaks, 'yo', linewidth=1.5)
plt.title("Detected Peaks in Hough Space", fontsize=FS)
plt.xlabel(r'$\theta$ (degrees)', fontsize=FS)
plt.ylabel(r'$\rho$', fontsize=FS)
plt.colorbar(label='Votes')
plt.tight_layout()
plt.savefig('Hough_Transform_Peaks.png')
plt.show()

# Tính góc trung vị để deskew
angle_median = np.rad2deg(np.median(a_peaks))
deskewed_img = rotate(Img, angle=90 + angle_median, resize=False, mode='edge')

# Hiển thị ảnh sau khi xoay (deskew)
plt.figure(3, figsize=(6, 6))
plt.imshow(deskewed_img, cmap='gray')
plt.title("Deskewed Image", fontsize=FS)
plt.axis('off')
plt.tight_layout()
plt.savefig('Deskewed_Paper_using_Hough_Transform.jpg')
plt.show()
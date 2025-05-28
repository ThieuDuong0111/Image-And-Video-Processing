import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, feature, img_as_ubyte
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.draw import circle_perimeter
from skimage.filters import prewitt
from skimage.feature import canny
from skimage import img_as_float
import cv2

# Đọc ảnh
image_path = 'coins.png'
Img = io.imread(image_path)

# Nếu ảnh màu thì chuyển sang xám
if Img.ndim == 3:
    gray = color.rgb2gray(Img)
else:
    gray = Img / 255.0

FS = 15  # fontsize
plt.figure(figsize=(15, 5))

# 1. Hiển thị ảnh gốc
plt.subplot(1, 3, 1)
plt.imshow(Img, cmap='gray')
plt.title("Original Image", fontsize=FS)
plt.axis('off')

# 2. Phát hiện biên bằng Prewitt
edges = prewitt(gray)
bw = edges > 0.15  # Ngưỡng bằng 0.15
plt.subplot(1, 3, 2)
plt.imshow(bw, cmap='gray')
plt.title("Edge Detection by Prewitt", fontsize=FS)
plt.axis('off')

# 3. Hough Transform để phát hiện hình tròn
# Chuyển về ảnh 8-bit nếu cần
bw_uint8 = img_as_ubyte(bw)

# Dải bán kính tìm kiếm
radii_range = np.arange(40, 201, 2)
hough_res = hough_circle(bw_uint8, radii_range)

# Lấy các vòng tròn mạnh nhất
accums, cx, cy, radii = hough_circle_peaks(hough_res, radii_range, total_num_peaks=10)

# Hiển thị kết quả
plt.subplot(1, 3, 3)
plt.imshow(Img, cmap='gray')
plt.title("Edge Detection by HT", fontsize=FS)
plt.axis('off')

# Vẽ tâm và đường tròn
for center_y, center_x, radius in zip(cy, cx, radii):
    circy, circx = circle_perimeter(center_y, center_x, radius)
    plt.plot(center_x, center_y, 'xr', markersize=2, linewidth=0.5)
    plt.plot(circx, circy, color='yellow', linewidth=0.25)

plt.tight_layout()
plt.savefig("Coin_Center_Findings_using_HT.jpg")
plt.show()
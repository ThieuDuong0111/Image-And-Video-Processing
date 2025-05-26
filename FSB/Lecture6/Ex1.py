import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from skimage.io import imread
from skimage import img_as_float
import os

# Load ảnh và chuyển về dạng float [0, 1]
Img = img_as_float(imread('street.jfif'))

# Khai báo 2 kernel làm nét
h1 = np.array([[0, -1, 0], [-1, 10, -1], [0, -1, 0]]) / 5
h2 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

# Hàm áp dụng kernel cho từng kênh màu (nếu ảnh màu)
def apply_filter(image, kernel, mode='reflect'):
    if image.ndim == 2:
        # Ảnh grayscale
        return convolve(image, kernel, mode=mode)
    elif image.ndim == 3:
        # Ảnh RGB
        filtered = np.zeros_like(image)
        for c in range(3):
            filtered[:, :, c] = convolve(image[:, :, c], kernel, mode=mode)
        return filtered
    else:
        raise ValueError("Unsupported image shape: {}".format(image.shape))

# Áp dụng hai bộ lọc làm nét
Img_sharpened1 = apply_filter(Img, h1, mode='reflect')     # giống 'symmetric' trong MATLAB
Img_sharpened2 = apply_filter(Img, h2, mode='nearest')     # giống 'replicate' trong MATLAB

# Hiển thị và lưu kết quả
FS = 15  # font size
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(Img)
axes[0].set_title("Original Image", fontsize=FS)
axes[1].imshow(Img_sharpened1)
axes[1].set_title("Sharpening 1", fontsize=FS)
axes[2].imshow(Img_sharpened2)
axes[2].set_title("Sharpening 2", fontsize=FS)

for ax in axes:
    ax.axis('off')

# Lưu hình ảnh
plt.tight_layout()
plt.savefig("Sharpening Images.jpg")
plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh xám
img = cv2.imread('moon.jpg', cv2.IMREAD_GRAYSCALE)
FS = 15  # font size

# Tính histogram ban đầu
count, bins = np.histogram(img.flatten(), bins=256, range=[0, 256])
max_count = np.max(count)

# Các tỉ lệ clipping
clip_ratios = [1, 0.7, 0.4, 0.05]
limited_eq_imgs = []
LUTs = np.zeros((len(clip_ratios), 256), dtype=np.uint8)

# Hàm tạo LUT từ histogram
def create_lut_from_hist(hist):
    cdf = np.cumsum(hist).astype(np.float64)
    cdf = (cdf - cdf.min()) / (cdf.max() - cdf.min())  # normalize
    lut = np.uint8(cdf * 255)
    return lut

# Áp dụng các tỷ lệ clip khác nhau
for i, ratio in enumerate(clip_ratios):
    clip = ratio * max_count
    clipped_count = np.clip(count, 0, clip)
    
    # Tạo ảnh 1 chiều ảo để tính LUT (giống MATLAB)
    clipped_img_1d = []
    for level in range(256):
        clipped_img_1d.extend([level] * int(clipped_count[level]))
    if len(clipped_img_1d) == 0:
        clipped_img_1d = [0]
    clipped_img_1d = np.array(clipped_img_1d, dtype=np.uint8)

    # Tạo LUT bằng histogram equalization
    lut = create_lut_from_hist(np.histogram(clipped_img_1d, bins=256, range=[0, 256])[0])
    LUTs[i] = lut

    # Tra LUT ra ảnh mới
    eq_img = cv2.LUT(img, lut)
    limited_eq_imgs.append(eq_img)

# Hiển thị các ảnh sau khi áp dụng CLAHE với các clip ratios khác nhau
plt.figure(1, figsize=(15, 5))
plt.clf()
for i, eq_img in enumerate(limited_eq_imgs):
    plt.subplot(1, len(clip_ratios), i+1)
    plt.imshow(eq_img, cmap='gray')
    plt.title(f'Clip at {clip_ratios[i]} max', fontsize=FS)
    plt.axis('off')
plt.tight_layout()
plt.savefig('CLAHE.png')

# Hiển thị histogram gốc và các LUT
plt.figure(2, figsize=(10, 4))
plt.show()



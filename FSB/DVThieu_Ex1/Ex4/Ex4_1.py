# Bài 1: Hiển thị ảnh gốc và histogram RGB
import cv2
import matplotlib.pyplot as plt

# Đọc ảnh màu và chuyển sang RGB để hiển thị đúng màu
img = cv2.imread('waterfall.jfif')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Tách các kênh
R, G, B = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]

# Hiển thị histogram và ảnh gốc
plt.figure(figsize=(10, 5))

# Histogram RGB
plt.subplot(1, 2, 1)
for channel, color in zip([R, G, B], ['r', 'g', 'b']):
    plt.hist(channel.ravel(), bins=256, range=(0, 256), color=color, alpha=0.5)
plt.xlabel('Index levels')
plt.ylabel('# pixels')
plt.title('Histogram of original image')

# Ảnh gốc
plt.subplot(1, 2, 2)
plt.imshow(img_rgb)
plt.title('Original Image')
plt.axis('off')

# Lưu hình
plt.tight_layout()
plt.savefig('Color_Histogram.jpeg')
plt.show()

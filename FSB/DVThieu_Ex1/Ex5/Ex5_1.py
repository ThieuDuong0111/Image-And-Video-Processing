import cv2
import matplotlib.pyplot as plt

# Đọc ảnh màu
image_color = cv2.imread('weather.png')

# Chuyển ảnh sang RGB để hiển thị đúng màu với matplotlib
image_rgb = cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB)

# Chuyển ảnh sang grayscale
image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

# Hiển thị ảnh gốc (color và grayscale)
plt.figure(figsize=(10, 5))

# Ảnh màu
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title('Original Color Image')
plt.axis('off')

# Ảnh grayscale
plt.subplot(1, 2, 2)
plt.imshow(image_gray, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')

plt.tight_layout()
plt.savefig('ex5_1.jpeg')
plt.show()

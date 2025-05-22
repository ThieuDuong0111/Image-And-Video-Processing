# Bài 3: Histogram Equalization trên kênh Value (HSV)
import cv2
import matplotlib.pyplot as plt

# Đọc ảnh và chuyển sang RGB
img = cv2.imread('waterfall.jfif')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Chuyển sang HSV
img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

# Tách kênh
H, S, V = cv2.split(img_hsv)

# Histogram equalization trên kênh V
V_eq = cv2.equalizeHist(V)

# Ghép lại HSV và chuyển sang RGB
hsv_eq = cv2.merge([H, S, V_eq])
img_eq_rgb = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2RGB)

# Hiển thị ảnh và histogram sau khi equalize
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.hist(cv2.cvtColor(img_eq_rgb, cv2.COLOR_RGB2GRAY).ravel(), bins=256, range=(0, 256), color='gray')
plt.title('Equalized Histogram')

plt.subplot(1, 2, 2)
plt.imshow(img_eq_rgb)
plt.title('Equalized Image (Histogram)')
plt.axis('off')

plt.tight_layout()
plt.savefig('Equalization_Histogram.jpeg')
plt.show()
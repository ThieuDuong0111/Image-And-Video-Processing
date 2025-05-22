import cv2
import matplotlib.pyplot as plt

# Đọc ảnh (bạn đổi 'image.jpg' thành tên file của bạn)
img = cv2.imread('apple.jpeg')

# Nếu ảnh là BGR, đổi sang RGB để hiển thị đúng màu
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# a. Flip left to right (tương đương fliplr)
img_fliplr = cv2.flip(img_rgb, 1)  # 1 là flip ngang (left-right)

# b. Rotate clockwise 180 degrees
# 180 độ = flip cả hai chiều (horizontal + vertical)
img_rot180 = cv2.rotate(img_rgb, cv2.ROTATE_180)

# c. Crop ½ trung tâm ảnh
h, w = img_rgb.shape[:2]
crop_w, crop_h = w // 2, h // 2
x_start = (w - crop_w) // 2
y_start = (h - crop_h) // 2
img_cropped = img_rgb[y_start:y_start+crop_h, x_start:x_start+crop_w]

# Hiển thị kết quả
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(img_fliplr)
plt.title("Flip Left-Right")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(img_rot180)
plt.title("Rotate 180° CW")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(img_cropped)
plt.title("Crop Center ½")
plt.axis('off')

plt.tight_layout()
plt.savefig('ex2_2.jpg') # Lưu ảnh kết quả
plt.show()

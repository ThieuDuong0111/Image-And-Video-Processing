import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh với OpenCV (theo mặc định là BGR)
image = cv2.imread('waterfall.jfif')

# Kiểm tra ảnh đã đọc thành công chưa
if image is None:
    raise ValueError("Ảnh không tồn tại hoặc đường dẫn không đúng.")

# Tách các kênh màu
B, G, R = cv2.split(image)

# Tạo ảnh chỉ chứa 1 kênh màu (với 2 kênh còn lại là 0)
zeros = np.zeros_like(B)
red_image = cv2.merge([zeros, zeros, R])
green_image = cv2.merge([zeros, G, zeros])
blue_image = cv2.merge([B, zeros, zeros])

# Trộn lại ảnh theo thứ tự BRG (không phải RGB)
brg_image = cv2.merge([B, R, G])

# Chuyển sang định dạng RGB để hiển thị đúng với matplotlib
red_image_rgb = cv2.cvtColor(red_image, cv2.COLOR_BGR2RGB)
green_image_rgb = cv2.cvtColor(green_image, cv2.COLOR_BGR2RGB)
blue_image_rgb = cv2.cvtColor(blue_image, cv2.COLOR_BGR2RGB)
brg_image_rgb = cv2.cvtColor(brg_image, cv2.COLOR_BGR2RGB)

# Hiển thị tất cả trên cùng một hình
plt.figure(figsize=(12, 6))

plt.subplot(1, 4, 1)
plt.imshow(red_image_rgb)
plt.title("Red Component")
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(green_image_rgb)
plt.title("Green Component")
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(blue_image_rgb)
plt.title("Blue Component")
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(brg_image_rgb)
plt.title("BRG Image")
plt.axis('off')

# Lưu ảnh
plt.tight_layout()
plt.savefig('Primary_Colors_and_BRG_Image.jpeg')
plt.show()
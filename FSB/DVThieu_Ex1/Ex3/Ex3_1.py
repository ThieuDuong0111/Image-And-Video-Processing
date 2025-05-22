import cv2
import matplotlib.pyplot as plt

# Đọc ảnh màu
cat_a_color = cv2.imread('cat_a.png')
cat_b_color = cv2.imread('cat_b.png')

if cat_a_color is None or cat_b_color is None:
    raise FileNotFoundError("Không tìm thấy file 'cat_a.png' hoặc 'cat_b.png'.")

# Chuyển sang ảnh RGB để hiển thị đúng màu trên matplotlib
cat_a_rgb = cv2.cvtColor(cat_a_color, cv2.COLOR_BGR2RGB)
cat_b_rgb = cv2.cvtColor(cat_b_color, cv2.COLOR_BGR2RGB)

# Chuyển sang ảnh xám
cat_a_gray = cv2.cvtColor(cat_a_color, cv2.COLOR_BGR2GRAY)
cat_b_gray = cv2.cvtColor(cat_b_color, cv2.COLOR_BGR2GRAY)

# Tạo figure
plt.figure(figsize=(10, 8))

# Ảnh màu cat_a
plt.subplot(2, 2, 1)
plt.imshow(cat_a_rgb)
plt.title("Color Image 1")
plt.axis('off')

# Ảnh màu cat_b
plt.subplot(2, 2, 2)
plt.imshow(cat_b_rgb)
plt.title("Color Image 2")
plt.axis('off')

# Ảnh xám cat_a
plt.subplot(2, 2, 3)
plt.imshow(cat_a_gray, cmap='gray')
plt.title("Grayscale Image 1")
plt.axis('off')

# Ảnh xám cat_b
plt.subplot(2, 2, 4)
plt.imshow(cat_b_gray, cmap='gray')
plt.title("Grayscale Image 2")
plt.axis('off')

plt.tight_layout()
plt.savefig("Color_Grayscale.jpg")  # Lưu hình tổng hợp
plt.show()

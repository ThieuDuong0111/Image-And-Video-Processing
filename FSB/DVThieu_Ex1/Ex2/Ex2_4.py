import cv2
import matplotlib.pyplot as plt

# Đọc ảnh xám
img = cv2.imread('apple.jpeg', cv2.IMREAD_GRAYSCALE)

# Khởi tạo CLAHE với clipLimit = 0.3
clahe = cv2.createCLAHE(clipLimit=0.3, tileGridSize=(8,8))
img_clahe = clahe.apply(img)

# Hiển thị ảnh gốc và ảnh sau CLAHE
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(img_clahe, cmap='gray')
plt.title('CLAHE (clipLimit = 0.3)')
plt.axis('off')

plt.tight_layout()
plt.savefig('ex2_4.jpg') # Lưu ảnh kết quả
plt.show()

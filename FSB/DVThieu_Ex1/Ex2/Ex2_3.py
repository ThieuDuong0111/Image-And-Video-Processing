import cv2
import matplotlib.pyplot as plt

# Đọc ảnh xám (nếu ảnh màu, chuyển sang xám)
img = cv2.imread('apple.jpeg', cv2.IMREAD_GRAYSCALE)

# Vẽ histogram ảnh gốc
plt.figure(figsize=(10,10))

plt.subplot(2,2,1)
plt.hist(img.ravel(), bins=256, range=[0,256], color='black')
plt.title('Histogram before equalization')
plt.xlabel('Gray Level')
plt.ylabel('# of pixels')

plt.subplot(2,2,2)
plt.imshow(img, cmap='gray')
plt.title('Image before equalization')
plt.axis('off')

# Cân bằng histogram toàn cục
img_eq = cv2.equalizeHist(img)

plt.subplot(2,2,3)
plt.hist(img_eq.ravel(), bins=256, range=[0,256], color='black')
plt.title('Histogram after equalization')
plt.xlabel('Gray Level')
plt.ylabel('# of pixels')

plt.subplot(2,2,4)
plt.imshow(img_eq, cmap='gray')
plt.title('Image after equalization')
plt.axis('off')

plt.tight_layout()
plt.savefig('ex2_3.jpg') # Lưu ảnh kết quả
plt.show()
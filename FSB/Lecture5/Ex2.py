import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load ảnh grayscale và chuyển sang float [0,1]
img = cv2.imread('bike.png', cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0
rows, cols = img.shape

# Tạo ma trận a, b, c
a = np.eye(rows // 2)
b = np.array([[1], [0]])
c = np.array([[0.5], [0.5]])

# Tính ma trận h1 và h2 bằng tích Kronecker
h1 = np.kron(a, b)
h2 = np.kron(a, c)

# Tính sub_Img1 và sub_Img2 bằng phép nhân ma trận
sub_img1 = h1.T @ img @ h1
sub_img2 = h2.T @ img @ h2

# Hiển thị ảnh
FS = 15
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image', fontsize=FS)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(sub_img1, cmap='gray')
plt.title('Method 1', fontsize=FS)
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(sub_img2, cmap='gray')
plt.title('Method 2', fontsize=FS)
plt.axis('off')

plt.tight_layout()
plt.savefig('Subsampling_2to1_Different_Methods.jpg')
plt.show()
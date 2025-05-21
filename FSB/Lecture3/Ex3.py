import cv2
import matplotlib.pyplot as plt

FS = 15  # fontsize for captions

# ==== (a) Cropping an image ====
# Uncomment và sửa đường dẫn nếu muốn chạy phần này
"""
img = cv2.imread('mandrill.tif')
if img is None:
    raise FileNotFoundError("Không tìm thấy file 'mandrill.tif'.")

# Crop vùng [xmin, ymin, width, height]
x1, x2 = 150, 450
y1, y2 = 100, 500
img_cropped = img[y1:y2, x1:x2]  # OpenCV: img[y1:y2, x1:x2]

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image', fontsize=FS)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB))
plt.title('Cropped Image', fontsize=FS)
plt.axis('off')

plt.tight_layout()
plt.savefig('Cropping Image.png')
plt.show()
"""

# ==== (b) Zooming an image with different interpolation methods ====

img = cv2.imread('bird.jpg')
if img is None:
    raise FileNotFoundError("Không tìm thấy file 'bird.jpg'.")

# Resize lên scale 4 (mặc định dùng INTER_CUBIC)
img_z1 = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
# Resize lên scale 3 với nội suy nearest neighbor (INTER_NEAREST)
img_z2 = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
# Resize lên scale 2 với nội suy bilinear (INTER_LINEAR)
img_z3 = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

plt.figure(figsize=(12, 10))
plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image', fontsize=FS)
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(cv2.cvtColor(img_z1, cv2.COLOR_BGR2RGB))
plt.title('Bicubic', fontsize=FS)
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(img_z2, cv2.COLOR_BGR2RGB))
plt.title('Nearest', fontsize=FS)
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(cv2.cvtColor(img_z3, cv2.COLOR_BGR2RGB))
plt.title('Bilinear', fontsize=FS)
plt.axis('off')

plt.tight_layout()
plt.savefig('Zooming_with_Different_Interpolation_Methods.png')
plt.show()

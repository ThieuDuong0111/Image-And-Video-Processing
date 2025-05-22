import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh màu và chuyển sang grayscale
img_bgr = cv2.imread('sea.jpg')

gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# Hàm thực hiện quantization theo số bit
def quantize(img, bits):
    levels = 2 ** bits
    quantized = np.floor(img / (256 / levels)) * (256 / levels)
    return quantized.astype(np.uint8)

# Tạo các ảnh đã được lượng tử hóa
img_2bit = quantize(gray, 2)
img_4bit = quantize(gray, 4)
img_6bit = quantize(gray, 6)
img_8bit = quantize(gray, 8)  # gần như giữ nguyên ảnh gốc

# Hiển thị các ảnh
titles = ['2-bits', '4-bits', '6-bits', '8-bits']
images = [img_2bit, img_4bit, img_6bit, img_8bit]

plt.figure(figsize=(12, 6))
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.imshow(images[i], cmap='gray', vmin=0, vmax=255)
    plt.title(titles[i], fontsize=10)
    plt.axis('off')

plt.tight_layout()
plt.savefig('ex4_sea_quantitize_grayscale.jpg') # Lưu ảnh kết quả
plt.show()

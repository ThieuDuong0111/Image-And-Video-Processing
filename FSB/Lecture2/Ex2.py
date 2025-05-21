import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh và chuyển sang float64 (tương đương double trong Octave)
img = cv2.imread('tiger.jpg', cv2.IMREAD_GRAYSCALE)
img = img.astype(np.float64)

# Thiết lập font size
FS = 15
plt.figure(figsize=(12, 6))
plt.tight_layout()

# Lặp qua các mức bit từ 1 đến 8
for num_bit in range(1, 9):
    num_level = 2 ** num_bit
    level_gap = 256 / num_level

    # Lượng tử hóa ảnh
    quantized_img = np.ceil(img / level_gap) * level_gap - 1
    quantized_img = np.clip(quantized_img, 0, 255)  # tránh giá trị vượt quá 255
    quantized_img = quantized_img.astype(np.uint8)

    # Hiển thị ảnh
    plt.subplot(2, 4, num_bit)
    plt.imshow(quantized_img, cmap='gray')
    title = f"{num_bit}-bit" if num_bit == 1 else f"{num_bit}-bits"
    plt.title(title, fontsize=FS)
    plt.axis('off')

    # Lưu ảnh
    filename = f"Quantization_in_{title}.jpg"
    cv2.imwrite(filename, quantized_img)

plt.suptitle("Quantization at Different Bit Depths", fontsize=FS+2)
plt.tight_layout()
plt.show()

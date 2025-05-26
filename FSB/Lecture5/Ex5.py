import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from skimage.io import imread
from mpl_toolkits.mplot3d import Axes3D

# Load ảnh
Img = imread('airplane.png', as_gray=True)  # load ảnh xám

# Tạo lưới tần số
wx, wy = np.meshgrid(np.linspace(-np.pi, np.pi, 33), np.linspace(-np.pi, np.pi, 33))

# Chọn loại filter: '2d', '1d_horizontal', hoặc '1d_vertical'
type = '2d'  # <== bạn có thể thay đổi ở đây

if type == '2d':
    H = (1/256) * (1 + 2*np.cos(wx) + 2*np.cos(2*wx)) * (1 + 2*np.cos(wy) + 2*np.cos(2*wy))
    h = (1/256) * np.ones((16, 16))  # spatial domain
elif type == '1d_horizontal':
    H = (1/16) * (1 + 2*np.cos(wx) + 2*np.cos(2*wx))
    h = (1/16) * np.ones((1, 16))  # 1D filter theo hàng
elif type == '1d_vertical':
    H = (1/16) * (1 + 2*np.cos(wy) + 2*np.cos(2*wy))
    h = (1/16) * np.ones((16, 1))  # 1D filter theo cột
else:
    raise ValueError("Unknown filter type")

# Vẽ tần số đáp ứng
FS = 15
fig = plt.figure(1, figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(wx/np.pi, wy/np.pi, np.abs(H), cmap='viridis')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(0, 1)
ax.set_xticks(np.arange(-1, 1.1, 0.5))
ax.set_yticks(np.arange(-1, 1.1, 0.5))
ax.set_xlabel(r'$\omega_x / \pi$')
ax.set_ylabel(r'$\omega_y / \pi$')
ax.set_zlabel(r'$|H(\omega_x, \omega_y)|$')
plt.tight_layout()
plt.savefig('Frequency_Responses_of_Three_Filters.jpg')
plt.show()

# Lọc ảnh sử dụng convolution
Filtered_Img = convolve(Img, h, mode='reflect')

# Hiển thị ảnh gốc và ảnh sau khi lọc
plt.figure(2, figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(Img, cmap='gray')
plt.title('Original Image', fontsize=FS)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(Filtered_Img, cmap='gray')
plt.title('Filtered Image', fontsize=FS)
plt.axis('off')

plt.tight_layout()
plt.savefig('Original_vs_Filtered_Images.jpg')
plt.show()
import cv2
import numpy as np
import matplotlib.pyplot as plt

kernel_radius = 16
kernel_sigma = 7
peak = 255
kernel_size = 2*kernel_radius + 1
c = cv2.getGaussianKernel(kernel_size,kernel_sigma)
cc = c.dot(c.T)
ccc = cc / np.max(cc)
cccc = peak * ccc
# print(ccc)

h, w = 512, 1024
img = np.full((h, w, 3), fill_value=8, dtype=np.uint8)

x = 127
y = 234

half = kernel_radius

# region of interest in the image
x0, x1 = max(0, x-half), min(w, x+half+1)
y0, y1 = max(0, y-half), min(h, y+half+1)

# corresponding region of the kernel
kx0, kx1 = half - (x - x0), half + (x1 - x)
ky0, ky1 = half - (y - y0), half + (y1 - y)

# add (with clipping) to all three channels
spot = cccc[ky0:ky1, kx0:kx1].astype(np.uint8)
for c in range(3):
    img[y0:y1, x0:x1, c] = np.clip(
        img[y0:y1, x0:x1, c].astype(int) + spot.astype(int),
        0, 255
    ).astype(np.uint8)



plt.figure()
plt.imshow(img)
# plt.clim(0, 1)
# plt.colorbar()
plt.show()
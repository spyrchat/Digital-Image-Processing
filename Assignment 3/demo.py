import numpy as np
from scipy.ndimage import convolve
from PIL import Image
import matplotlib.pyplot as plt
from wiener_filtering import my_wiener_filter
import hw3_helper_utils  # Ensure this module is in the same directory or in the Python path

# Load the image and convert it to a grayscale NumPy array
filename = "Assignment 3/cameraman.tif"
img = Image.open(filename)

bw_img = img.convert("L")
img_array = np.array(bw_img)

x = img_array / 255.0  # Normalize the image
# Create white noise with level 0.02
v = 0.02 * np.random.randn(*x.shape)
# Create motion blur filter
h = hw3_helper_utils.create_motion_blur_filter(length=20, angle=30)
# Obtain the filtered image
y0 = convolve(x, h, mode="wrap")
# Generate the noisy image
y = y0 + v

# Wiener filter with trial and error for K
K = 0.01  # Example K value, adjust as needed
x_hat = my_wiener_filter(y, h, K)

# Inverse filtering for noiseless image y0
x_inv0 = my_wiener_filter(y0, h, K)

# Fourier domain inverse filter approach
H = np.fft.fft2(h, s=y.shape)
H_inv = np.conj(H) / (np.abs(H) ** 2 + K)
Y = np.fft.fft2(y)
X_inv = H_inv * Y
x_inv = np.fft.ifft2(X_inv).real

# Display the results
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

axs[0][0].imshow(x, cmap='gray')
axs[0][0].set_title("Original image x")
axs[0][0].axis('off')

axs[0][1].imshow(y0, cmap='gray')
axs[0][1].set_title("Clean image y0")
axs[0][1].axis('off')

axs[0][2].imshow(y, cmap='gray')
axs[0][2].set_title("Blurred and noisy image y")
axs[0][2].axis('off')

axs[1][0].imshow(x_inv0, cmap='gray')
axs[1][0].set_title("Inverse filtering noiseless output x_inv0")
axs[1][0].axis('off')

axs[1][1].imshow(x_inv, cmap='gray')
axs[1][1].set_title("Inverse filtering noisy output x_inv")
axs[1][1].axis('off')

axs[1][2].imshow(x_hat, cmap='gray')
axs[1][2].set_title("Wiener filtering output x_hat")
axs[1][2].axis('off')

plt.tight_layout()
plt.show()

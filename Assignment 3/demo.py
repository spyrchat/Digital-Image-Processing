import numpy as np
from scipy.ndimage import convolve
from PIL import Image
import matplotlib.pyplot as plt
from wiener_filtering import my_wiener_filter, inverse_H_filter
import hw3_helper_utils  

# Load the image and convert it to a grayscale NumPy array
filename = "Assignment 3/cameraman.tif"
img = Image.open(filename)
bw_img = img.convert("L")
img_array = np.array(bw_img)
x = img_array / 255.0  # Normalize the image
# Create white noise
v = 0.2 * np.random.randn(*x.shape)
# Create motion blur filter
h = hw3_helper_utils.create_motion_blur_filter(length=10, angle=0) 
# Obtain the filtered image
y0 = convolve(x, h, mode="wrap")
# Generate the noisy image
y = y0 + v

x_inv0 = inverse_H_filter(y0, h)

# Define a range of K values for the grid search
K_values = np.logspace(-3, 10, num=300)  # Logarithmically spaced values between 0.001 and 100
# Initialize variables to store the best K and the corresponding minimum MSE
best_K = None
min_mse = float('inf')

# Initialize variables to store the best K and the corresponding minimum MSE
best_K = None
min_mse = float('inf')

# Perform a grid search in order to find the optimal K value
J_values = []
for K in K_values:
    x_hat = my_wiener_filter(y, h, K)
    mse = np.mean((x_inv0 - x_hat)**2)
    J_values.append(mse)
    if mse < min_mse:
        min_mse = mse
        best_K = K

print(f"Optimal K: {best_K}")
print(f"Minimum MSE: {min_mse}")

#------------- Plot the J(K) curve -------------#
# Define a finer range of K values around the optimal K
K_fine_values = np.linspace(best_K / 10, best_K * 10, num=100)
J_fine_values = []
for K in K_fine_values:
    x_hat = my_wiener_filter(y, h, K)
    mse = np.mean((x_inv0 - x_hat)**2)
    J_fine_values.append(mse)

# Plot the J(K) curve for the finer range around the optimal K
plt.figure(figsize=(10, 6))
plt.plot(K_fine_values, J_fine_values, marker='o')
plt.axvline(best_K, color='r', linestyle='--', label=f'Optimal K = {best_K:.4f}')
plt.xlabel('K')
plt.ylabel('J(K)')
plt.xscale('log')
plt.yscale('log')
plt.title('J(K) vs K around optimal K')
plt.legend()
plt.grid(True)
plt.show()

x_hat = my_wiener_filter(y, h, best_K)

x_inv = inverse_H_filter(y, h)

#------------- Display the results -------------#
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

axs[0][0].imshow(np.clip(x,0,1), cmap='gray')
axs[0][0].set_title("Original image x")
axs[0][0].axis('off')

axs[0][1].imshow(np.clip(y0,0,1), cmap='gray')
axs[0][1].set_title("Clean image y0")
axs[0][1].axis('off')

axs[0][2].imshow(np.clip(y,0,1), cmap='gray')
axs[0][2].set_title("Blurred and noisy image y")
axs[0][2].axis('off')

axs[1][0].imshow(np.clip(x_inv0,0,1), cmap='gray')
axs[1][0].set_title("Inverse filtering noiseless output x_inv0")
axs[1][0].axis('off')

axs[1][1].imshow(np.clip(x_inv,0,1), cmap='gray')
axs[1][1].set_title("Inverse filtering noisy output x_inv")
axs[1][1].axis('off')

axs[1][2].imshow(np.clip(x_hat,0,1), cmap='gray')
axs[1][2].set_title("Wiener filtering output x_hat")
axs[1][2].axis('off')

plt.tight_layout()
plt.show()

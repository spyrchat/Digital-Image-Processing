import numpy as np
import cv2
import matplotlib.pyplot as plt

def my_hough_transform(img_grayscale: np.ndarray, d_rho: int, d_theta: float, n: int):
    # Step 1: Perform edge detection using Canny edge detector from OpenCV
    edges = cv2.Canny(img_grayscale, 50, 150)
    print("Edge detection completed.")

    # Image dimensions
    height, width = edges.shape
    
    # Step 2: Define the range of rho and theta
    max_rho = int(np.hypot(height, width))  # Maximum possible value of rho
    thetas = np.arange(-np.pi / 2, np.pi / 2, d_theta)  # Range of theta values
    rhos = np.arange(-max_rho, max_rho, d_rho)  # Range of rho values
    
    # Step 3: Initialize the accumulator array
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=int)
    print("Accumulator array initialized.")

    # Step 4: Get the indices of the edge points
    y_idxs, x_idxs = np.nonzero(edges)  # Get the coordinates of edge points

    # Precompute cosine and sine of thetas
    cos_thetas = np.cos(thetas)
    sin_thetas = np.sin(thetas)
    
    # Step 5: Populate the accumulator array using vectorized operations
    for y, x in zip(y_idxs, x_idxs):
        rho_vals = x * cos_thetas + y * sin_thetas
        rho_indices = np.round((rho_vals - rhos[0]) / d_rho).astype(int)
        valid_indices = (rho_indices >= 0) & (rho_indices < len(rhos))
        accumulator[rho_indices[valid_indices], np.arange(len(thetas))[valid_indices]] += 1
    
    print("Accumulator array populated.")

    # Step 6: Find the n highest peaks in the accumulator array
    flat_indices = np.argpartition(accumulator.ravel(), -n)[-n:]
    peak_indices = np.column_stack(np.unravel_index(flat_indices, accumulator.shape))
    
    # Extract the rho and theta values for the strongest lines
    rho_theta_pairs = [(rhos[rho_idx], thetas[theta_idx]) for rho_idx, theta_idx in peak_indices]
    print("Peaks detected in the accumulator array.")
    print("Peak indices:", peak_indices)

    # Step 7: Create the output parameter list for the strongest lines
    L = np.array(rho_theta_pairs)
    
    return accumulator, L, edges

def draw_lines_on_image(image, lines, color=(0, 255, 0), thickness=2):
    height, width = image.shape[:2]
    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        # Extend the line by 1000 pixels in both directions to ensure it covers the entire image
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(image, (x1, y1), (x2, y2), color, thickness)

# Load the grayscale image
img_path = 'Assignment 2/im2.jpg'
img_grayscale = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Check if the image is loaded correctly
if img_grayscale is None:
    raise FileNotFoundError("The image file could not be loaded. Please check the path and the file.")
print("Image loaded successfully.")

# Optionally reduce the resolution to speed up processing
img_grayscale = cv2.resize(img_grayscale, (img_grayscale.shape[1] // 8, img_grayscale.shape[0] // 8))

# Parameters
d_rho = 1
d_theta = np.pi / 180
n = 10

# Perform Hough Transform
H, L, edges = my_hough_transform(img_grayscale, d_rho, d_theta, n)

# Plot the Hough Transform accumulator array
plt.figure(figsize=(10, 10))
plt.imshow(H, cmap='gray')
plt.title('Hough Transform Accumulator')
plt.xlabel('Theta (radians)')
plt.ylabel('Rho (pixels)')
plt.show()

# Display the original image with edge points highlighted
y_idxs, x_idxs = np.nonzero(edges)
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(img_grayscale, cv2.COLOR_GRAY2RGB))
plt.scatter(x_idxs, y_idxs, color='lightgray', s=0.5)  # Highlight edge points
plt.title('Original Image with Edge Points')
plt.show()

# Draw the detected lines on the original image
img_with_lines = cv2.cvtColor(img_grayscale, cv2.COLOR_GRAY2BGR)
draw_lines_on_image(img_with_lines, L)

# Display the original image with detected lines
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(img_with_lines, cv2.COLOR_BGR2RGB))
plt.title('Detected Lines on Original Image')
plt.show()

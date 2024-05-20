import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import feature
import math
from PIL import Image
def my_hough_transform(img_binary: np.ndarray, d_rho: int, d_theta: float, n: int):
    # Image dimensions
    height,width = np.shape(img_binary)
    # Step 2: Define the range of rho and theta
    max_rho = int(np.hypot(height, width))  # Maximum possible value of rho
    thetas = np.arange(-np.pi / 2, np.pi / 2, d_theta)  # Range of theta values
    rhos = np.arange(-max_rho, max_rho, d_rho)  # Range of rho values
    
    # Step 3: Initialize the accumulator array
    H = np.zeros((len(rhos), len(thetas)), dtype=int)
    print("Accumulator array initialized.")
    
    # Step 5: Populate the accumulator array using vectorized operations
    for x in range(width):
            for y in range(height):
                if img_binary[y, x] == 1:
                    for j in range(len(thetas)-1):
                        theta = (thetas[j] + thetas[j+1]) /2
                        rho = x * math.cos(theta) + y * math.sin(theta)
                        #if 0<= rho <2*max_rho:
                        rho_idx = int((rho+max_rho) / d_rho)
                        H[rho_idx, j] += 1
    
    print("Accumulator array populated.")

    # Step 6: Find the n highest peaks in the accumulator array
    flat_indices = np.argpartition(H.ravel(), -n)[-n:]
    peak_indices = np.column_stack(np.unravel_index(flat_indices, H.shape))
    
    # Extract the rho and theta values for the strongest lines
    rho_theta_pairs = [(rhos[rho_idx], thetas[theta_idx]) for rho_idx, theta_idx in peak_indices]
    print("Peaks detected in the accumulator array.")
    print("Peak indices:", peak_indices)

    # Step 7: Create the output parameter list for the strongest lines
    L = np.array(rho_theta_pairs)
    
    # Count the number of points not part of the n strongest lines
    res = np.sum(img_binary) - np.sum(H)
    
    return H, L, res

def draw_lines_on_image(image, lines, color=(0, 255, 0), thickness=2):
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

if __name__ == "__main__":
    # Load the grayscale image
    img_path = 'Assignment 2/im3.jpg'
    img = Image.open(fp=img_path)
    img = img.resize((215, 360))
    # Keep only the Luminance component of the image
    img_grayscale = img.convert("L")
    img_grayscale = np.array(img_grayscale)
    # Perform edge detection using Canny edge detector from OpenCV
    print("Edge detection completed.")

    # Convert the edge-detected image to binary using thresholding
    img_canny = feature.canny(img_grayscale, sigma=4, low_threshold=10, high_threshold=20)

    # Parameters
    d_rho = 1
    d_theta = np.pi / 180
    n = 20
    max_rho = int(np.hypot(img_grayscale.shape[0], img_grayscale.shape[1]))
    thetas = np.arange(-np.pi / 2, np.pi / 2, d_theta)
    rhos = np.arange(-max_rho, max_rho, d_rho)
    # Perform Hough Transform
    H, L, res = my_hough_transform(img_canny, d_rho, d_theta, n)
    print("H: ",H)
    # Plot the Hough Transform accumulator array and highlight the peaks
    plt.figure(figsize=(10, 10))
    plt.imshow(H, cmap='gray', extent=[np.rad2deg(thetas[0]), np.rad2deg(thetas[-1]), rhos[-1], rhos[0]], aspect='auto')
    plt.scatter(np.rad2deg(L[:, 1]), L[:, 0], color='red', s=50, edgecolors='yellow')  # Highlight peaks
    plt.title('Hough Transform Accumulator')
    plt.xlabel('Theta (degrees)')
    plt.ylabel('Rho (pixels)')
    plt.show()
    # Display the original image with edge points highlighted

    y_idxs, x_idxs = np.nonzero(img_canny)
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img_grayscale, cv2.COLOR_GRAY2RGB))
    plt.scatter(x_idxs, y_idxs, color='red', s=1)  # Highlight edge points
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

    print(f"Number of points not part of the {n} strongest lines: {res}")

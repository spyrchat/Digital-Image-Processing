import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import feature
import math
from PIL import Image

def my_hough_transform(img_binary: np.ndarray, d_rho: int, d_theta: float, n: int, method='Fast'):
    # Image dimensions
    height, width = np.shape(img_binary)
    # Rho max is the hypotenuse of the triangle that is formed with height and width as vertical lines
    max_rho = int(np.hypot(height, width))
    thetas = np.arange(-np.pi / 2, np.pi / 2, d_theta)  # Range of theta values
    rhos = np.arange(-max_rho, max_rho, d_rho)  # Range of rho values
    
    if method == 'Fast':
        H = np.zeros((len(rhos), len(thetas)), dtype=int)
        # Get the indices of the binary image where the pixel value is 1
        y_idxs, x_idxs = np.nonzero(img_binary)
        # Compute the midpoints of the thetas
        theta_midpoints = (thetas[:-1] + thetas[1:]) / 2
        # Compute the cosines and sines of the theta midpoints
        cos_thetas = np.cos(theta_midpoints)
        sin_thetas = np.sin(theta_midpoints)
        # Calculate the rho values for all (x, y) pairs and theta midpoints
        rho_values = np.outer(x_idxs, cos_thetas) + np.outer(y_idxs, sin_thetas)
        # Quantize the rho values to get the corresponding rho indices
        rho_indices = ((rho_values + max_rho) / d_rho).astype(int)
        # Accumulate the votes in the accumulator array H
        for i in range(len(theta_midpoints)):
            np.add.at(H[:, i], rho_indices[:, i], 1)
        # Find the n highest peaks in the accumulator array
        indices = np.argpartition(H.flatten(), -2)[-n:]
        L_rhos_idxes, L_theta_idxes = np.unravel_index(indices, H.shape)
        L = np.vstack((rhos[L_rhos_idxes], thetas[L_theta_idxes])).T
        img_pixels = height * width
        # Count the number of points not part of the n strongest lines
        res = img_pixels - np.sum(H)

        # Compute res
        # Iterate over each pixel in the image to find how many pixels are in the lines of L
        num_of_pixels_in_lines = 0
        for y in range(height):
            for x in range(width):
                # Check if the pixel (i, j) is in any of the lines
                for rho, theta in L:
                    # Small tolerance if the pixel is in line
                    if abs(rho - (y * np.cos(theta) + y * np.sin(theta))) < 1:  
                        num_of_pixels_in_lines += 1
                        break

        res = height * width - num_of_pixels_in_lines
        return H, L, res
    
    if method == 'ByTheBook':
        H = np.zeros((len(rhos), len(thetas)), dtype=int)
        # Accumulate votes in H
        for x in range(width):
            for y in range(height):
                if img_binary[y, x] == 1:
                    for j in range(len(thetas) - 1):
                        theta = (thetas[j] + thetas[j + 1]) / 2
                        rho = x * math.cos(theta) + y * math.sin(theta)
                        rho_idx = int((rho + max_rho) / d_rho)
                        H[rho_idx, j] += 1

        # Find the n highest peaks in the accumulator array
        flat_indices = np.argpartition(H.ravel(), -2)[-n:]
        peak_indices = np.column_stack(np.unravel_index(flat_indices, H.shape))
        
        # Extract the rho and theta values for the strongest lines
        rho_theta_pairs = [(rhos[rho_idx], thetas[theta_idx]) for rho_idx, theta_idx in peak_indices]
        # Create the output parameter list for the strongest lines
        L = np.array(rho_theta_pairs)
        
        num_of_pixels_in_lines = 0
        for y in range(height):
            for x in range(width):
                # Check if the pixel (i, j) is in any of the lines
                for rho, theta in L:
                    # Small tolerance if the pixel is in line
                    if abs(rho - (y * np.cos(theta) + y * np.sin(theta))) < 1:  
                        num_of_pixels_in_lines += 1
                        break

        res = height * width - num_of_pixels_in_lines
        return H, L, res

def draw_lines_on_image(image, lines, scale_x=1, scale_y=1, color=(0, 255, 0), thickness=2):
    # Iterate through each element in L to find the line equations that are represented in each cell of L
    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho * scale_x
        y0 = b * rho * scale_y
        # Extend the line by 1000 pixels in both directions to ensure it covers the entire image
        x1 = int(x0 + 10000 * (-b))
        y1 = int(y0 + 10000 * (a))
        x2 = int(x0 - 10000 * (-b))
        y2 = int(y0 - 10000 * (a))

        cv2.line(image, (x1, y1), (x2, y2), color, thickness)

if __name__ == "__main__":
    img_path = 'Assignment 2/im5.jpg'
    img = Image.open(fp=img_path)
    img_high_res_rgb = np.array(img)
    img_high_res = img.convert("L")
    img_high_res = np.array(img_high_res)
    height_high, width_high = img_high_res.shape
    
    # Resize the image for lower resolution processing
    scale_factor = 0.1  # Scale factor for lower resolution
    img_low_res = cv2.resize(img_high_res, (int(width_high * scale_factor), int(height_high * scale_factor)))
    img_low_res_rgb = cv2.resize(img_high_res_rgb, (int(width_high * scale_factor), int(height_high * scale_factor)))
    height_low, width_low = img_low_res.shape

    # Perform edge detection using Canny edge detector
    img_canny = feature.canny(img_low_res, sigma=4, low_threshold=2, high_threshold=10)

    # Parameters for Hough Transform
    d_rho = 1
    d_theta = np.pi / 360
    n = 30
    max_rho = int(np.hypot(height_low, width_low))
    thetas = np.arange(-np.pi / 2, np.pi / 2, d_theta)
    rhos = np.arange(-max_rho, max_rho, d_rho)

    # Perform Hough Transform
    H, L, res = my_hough_transform(img_canny, d_rho, d_theta, n)

    # Plot the Hough Transform accumulator array and highlight the peaks
    plt.figure(figsize=(10, 10))
    plt.imshow(H, cmap='gray', extent=[np.rad2deg(thetas[0]), np.rad2deg(thetas[-1]), rhos[-1], rhos[0]], aspect='auto')
    plt.scatter(np.rad2deg(L[:, 1]), L[:, 0], color='red', s=50, edgecolors='yellow')  # Highlight peaks
    plt.title('Hough Transform Accumulator')
    plt.xlabel('Theta (degrees)')
    plt.ylabel('Rho (pixels)')
    plt.show()

    # Display the low-resolution RGB image with edge points highlighted
    y_idxs, x_idxs = np.nonzero(img_canny)
    plt.figure(figsize=(10, 10))
    plt.imshow(img_low_res_rgb)
    plt.scatter(x_idxs, y_idxs, color='red', s=1)  # Highlight edge points
    plt.title('Low-resolution Image with Edge Points')
    plt.show()

    # Draw the detected lines on the high-resolution RGB image
    img_with_lines = img_high_res_rgb.copy()
    draw_lines_on_image(img_with_lines, L, scale_x=1/scale_factor, scale_y=1/scale_factor)

    # Display the high-resolution RGB image with detected lines
    plt.figure(figsize=(10, 10))
    plt.imshow(img_with_lines)
    plt.title('Detected Lines on High-resolution Image')
    plt.show()

    print(f"Number of points not part of the {n} strongest lines: {res}")
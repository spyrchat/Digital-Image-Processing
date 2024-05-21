import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import feature
from PIL import Image
from deliverable_1 import my_hough_transform
from deliverable_2 import my_corner_harris, my_corner_peaks
from sklearn.cluster import DBSCAN
from itertools import combinations
import math

# Load and preprocess the image
img_path = 'Assignment 2/im1.jpg'
img = Image.open(fp=img_path)
img = img.resize((215, 360))

# Convert the image to grayscale
img_grayscale = img.convert("L")
img_grayscale = np.array(img_grayscale)

# Perform edge detection using Canny edge detector from skimage
img_canny = feature.canny(img_grayscale, sigma=4, low_threshold=10, high_threshold=20)
print("Edge detection completed.")

# Perform Harris corner detection and find corner peaks
R = my_corner_harris(img_grayscale / 255.0, k=0.04, sigma=2.0)
corners = my_corner_peaks(R, rel_threshold=0.01)

# Parameters for Hough Transform
d_rho = 1
d_theta = np.pi / 180
n = 20

# Perform Hough Transform
H, L, res = my_hough_transform(img_canny, d_rho, d_theta, n)
print("L: ", L)
# Function to convert polar coordinates to Cartesian coordinates
def polar_to_cartesian(rho, theta):
    A = np.cos(theta)
    B = np.sin(theta)
    C = -rho
    return A, B, C

# Convert lines from polar to Cartesian
def convert_lines_polar_to_cartesian(lines):
    cartesian_lines = []
    for rho, theta in lines:
        A, B, C = polar_to_cartesian(rho, theta)
        cartesian_lines.append((A, B, C))
    return cartesian_lines

# Find the intersection of two lines
def find_intersection(line1, line2):
    A1, B1, C1 = line1
    A2, B2, C2 = line2

    # Coefficient matrix A
    A = np.array([[A1, B1], [A2, B2]])
    # Constant matrix B
    B = np.array([-C1, -C2])

    # Check if determinant is zero
    if np.linalg.det(A) == 0:
        return None
    else:
        # Solve the system of equations
        intersection_point = np.linalg.solve(A, B)
        return intersection_point
def are_perpendicular(theta1, theta2):
    """
    Check if two lines in polar coordinates are perpendicular.

    Parameters:
    theta1 (float): The angle of the first line in radians.
    theta2 (float): The angle of the second line in radians.

    Returns:
    bool: True if the lines are perpendicular, False otherwise.
    """
    # Normalize the angles to be within the range [0, 2*pi)
    theta1 = theta1 % (2 * math.pi)
    theta2 = theta2 % (2 * math.pi)
    
    # Compute the absolute difference between the angles
    diff = abs(theta1 - theta2)
    
    # Check if the difference is pi/2 or 3*pi/2
    return math.isclose(diff, math.pi / 2) or math.isclose(diff, 3 * math.pi / 2)

# Convert detected lines from polar to Cartesian
lines = convert_lines_polar_to_cartesian(L)

intersections = []
for i in range(len(lines)):
    for j in range(i + 1, len(lines)):  # Ensure j > i to avoid duplicate checks
        # Check if the angle difference or its complement is within the specified bounds
        if are_perpendicular(L[i][1], L[j][1]):
            print(f"Lines {i} and {j} are close to perpendicular.")
            temp = find_intersection(lines[i], lines[j])
            if temp is not None:
                # Floor the intersection point and convert to integers
                temp = np.floor(temp).astype(int)
                intersections.append(temp)

print(f"Intersections: {intersections}")
        

print("Intersections: ", intersections)
H,W = np.shape(img_grayscale)
normalized_intersections = []
for point in intersections:
    x, y = point
    # Scale x and y to fit within image dimensions
    x_norm = np.clip(x, 0, W - 1)
    y_norm = np.clip(y, 0, H - 1)
    normalized_intersections.append((x_norm, y_norm))

normalized_intersections = np.array(normalized_intersections)

print("Normalized Intersections: ", normalized_intersections)

# List to store filtered intersections
filtered_intersections = []

print("Corners :", corners)

# Calculate distance and filter intersections
for intersection in normalized_intersections:
    for corner in corners:
        distance = np.sqrt((intersection[0] - corner[1])**2 + (intersection[1] - corner[0])**2)
        if distance < 2.5:  # Adjust the distance threshold if necessary
            filtered_intersections.append(corner.astype(int))

filtered_intersections = np.array(filtered_intersections)

# Define a function to remove points that are too close to each other
def remove_close_points(points, min_distance):
    filtered_points = []
    for point in points:
        if all(np.linalg.norm(point - np.array(fp)) >= min_distance for fp in filtered_points):
            filtered_points.append(point)
    return np.array(filtered_points)

# Define the minimum distance threshold
min_distance = 10  # Adjust the distance threshold to 5 pixels

# Sort points for consistent order
filtered_intersections = filtered_intersections[np.lexsort((filtered_intersections[:, 1], filtered_intersections[:, 0]))]

# Remove close points from filtered_intersections
unique_filtered_intersections = remove_close_points(filtered_intersections, min_distance)

print("Filtered Intersections (after removal):", unique_filtered_intersections)


plt.figure(figsize=(10, 8))
plt.imshow(img, cmap='gray')
for point in unique_filtered_intersections:
    plt.plot(point[1], point[0], 'bo')  # 'bo' for blue circles

plt.figure(figsize=(10, 8))
plt.imshow(img, cmap='gray')
for point in corners:
    plt.plot(point[1], point[0], 'ro')  # 'ro' for red circles
plt.title('Corners')
plt.show()
img_array = np.array(img)


# Identify the highest rightmost point
highest_rightmost = unique_filtered_intersections[np.argmax(unique_filtered_intersections[:, 0] * 1000 + unique_filtered_intersections[:, 1])]
# Identify the highest leftmost point
highest_leftmost = unique_filtered_intersections[np.argmax(unique_filtered_intersections[:, 0] * 1000 - unique_filtered_intersections[:, 1])]
# Identify the lowest rightmost point
lowest_rightmost = unique_filtered_intersections[np.argmin(unique_filtered_intersections[:, 0] * 1000 - unique_filtered_intersections[:, 1])]
# Identify the lowest leftmost point
lowest_leftmost = unique_filtered_intersections[np.argmin(unique_filtered_intersections[:, 0] * 1000 + unique_filtered_intersections[:, 1])]

print("Highest rightmost point:", highest_rightmost)
print("Highest leftmost point:", highest_leftmost)
print("Lowest rightmost point:", lowest_rightmost)
print("Lowest leftmost point:", lowest_leftmost)

# Plot the points on the original image
plt.figure()
plt.imshow(img_array, cmap='gray')
plt.scatter([highest_rightmost[1], highest_leftmost[1], lowest_rightmost[1], lowest_leftmost[1]],
            [highest_rightmost[0], highest_leftmost[0], lowest_rightmost[0], lowest_leftmost[0]],
            c='red', s=100, label='Key Points')
plt.legend()
plt.title('Original Image with Key Points')
plt.axis('off')
plt.show()


import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import DBSCAN
from deliverable_1 import my_hough_transform
from fast_harris import my_corner_harris, my_corner_peaks
from skimage import feature
# Load and preprocess the image
img_path = 'Assignment 2/im3.jpg'
img = Image.open(fp=img_path)
img = img.resize((510, 660))

# Convert the image to grayscale
img_grayscale = img.convert("L")
img_grayscale = np.array(img_grayscale)

# Perform edge detection using Canny edge detector from skimage
img_canny = feature.canny(img_grayscale, sigma=4, low_threshold=10, high_threshold=20)
print("Edge detection completed.")

# Perform Harris corner detection and find corner peaks
R = my_corner_harris(img_grayscale, k=0.04, sigma=2.0)
corners = my_corner_peaks(R, rel_threshold=0.005)

# Parameters for Hough Transform
d_rho = 1
d_theta = np.pi / 180
n = 30

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

# Convert detected lines from polar to Cartesian
lines = convert_lines_polar_to_cartesian(L)

# List to store intersection points
intersections = []

# Find intersections between all pairs of lines
for i in range(len(lines)):
    for j in range(i + 1, len(lines)):  # Ensure j > i to avoid duplicate checks
        temp = find_intersection(lines[i], lines[j])
        if temp is not None:
            # Floor the intersection point and convert to integers
            temp = np.floor(temp).astype(int)
            intersections.append(temp)

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
        distance = np.sqrt((intersection[0] - corner[0])**2 + (intersection[1] - corner[1])**2)
        if distance < 10:  # Adjust the distance threshold if necessary
            filtered_intersections.append(intersection.astype(int))

print("Filtered Intersections :", filtered_intersections)

def remove_close_points(points, min_distance):
    if not points:
        return []

    # Sort points for consistent order
    points = sorted(points, key=lambda x: (x[0], x[1]))
    unique_points = [points[0]]

    for point in points[1:]:
        if all(np.linalg.norm(point - np.array(p)) >= min_distance for p in unique_points):
            unique_points.append(point)

    return unique_points

# Define the minimum distance threshold
min_distance = 10

# Remove close points from filtered intersections
unique_filtered_intersections = remove_close_points(filtered_intersections, min_distance)

print("Filtered Intersections (after removal):", unique_filtered_intersections)


plt.figure(figsize=(10, 8))
plt.imshow(img, cmap='gray')
for point in unique_filtered_intersections:
    plt.plot(point[0], point[1], 'ro')  # 'ro' for red circles
plt.title('Filtered Intersections on Original Image')
plt.show()
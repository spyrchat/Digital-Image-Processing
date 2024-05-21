import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import DBSCAN
from deliverable_1 import my_hough_transform
from fast_harris import my_corner_harris, my_corner_peaks
from skimage import feature

def find_intersections(L):
    intersections = []
    for i in range(len(L)):
        for j in range(i + 1, len(L)):
            rho1, theta1 = L[i]
            rho2, theta2 = L[j]

            # Check if lines are perpendicular
            if np.abs(np.abs(theta1 - theta2) - np.pi / 2) < 1e-2:
                intersection = find_intersection(rho1, theta1, rho2, theta2)
                intersections.append(intersection)
    return np.array(intersections)

# Function to find the intersection of two lines given their (rho, theta) parameters
def find_intersection(rho1, theta1, rho2, theta2):
    A1 = np.cos(theta1)
    B1 = np.sin(theta1)
    C1 = rho1

    A2 = np.cos(theta2)
    B2 = np.sin(theta2)
    C2 = rho2

    A = np.array([[A1, B1], [A2, B2]])
    B = np.array([C1, C2])

    intersection_point = np.linalg.solve(A, B)

    return intersection_point


if __name__ == "__main__":
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
    n = 11

    # Perform Hough Transform
    H, L, res = my_hough_transform(img_canny, d_rho, d_theta, n)


    intersections = find_intersections(L)
    intersections = np.floor(intersections).astype(int)
    plt.figure(figsize=(10, 8))
    plt.imshow(img, cmap='gray')
    for point in intersections:
        plt.plot(point[1], point[0], 'bo')  # 'bo' for blue circles

    # Calculate distance and filter intersections
    for intersection in intersections:
        for corner in corners:
            distance = np.sqrt((intersection[0] - corner[0])**2 + (intersection[1] - corner[1])**2)
            if distance < 2.5:  # Adjust the distance threshold if necessary
                intersections.append(corner.astype(int))

    filtered_intersections = np.array(intersections)
    print(filtered_intersections)



    plt.figure(figsize=(10, 8))
    plt.imshow(img, cmap='gray')
    for point in corners:
        plt.plot(point[1], point[0], 'ro')  # 'ro' for red circles
    plt.title('Corners')
    plt.show()
    img_array = np.array(img)
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import feature
from PIL import Image
from deliverable_1 import my_hough_transform
from deliverable_2 import my_corner_harris, my_corner_peaks
from deliverable_3 import my_img_rotation
from itertools import combinations
import math
import os
from sklearn.cluster import DBSCAN

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

def are_perpendicular(theta1, theta2, tol_deg=3):
    tol_rad = math.radians(tol_deg)  # Convert tolerance from degrees to radians
    # Normalize the angles to be within the range [0, 2*pi)
    theta1 = theta1 % (2 * math.pi)
    theta2 = theta2 % (2 * math.pi)
    # Compute the absolute difference between the angles
    diff = abs(theta1 - theta2)
    # Check if the difference is close to pi/2 or 3*pi/2 within the tolerance range
    return (math.isclose(diff, math.pi / 2, abs_tol=tol_rad) or 
            math.isclose(diff, 3 * math.pi / 2, abs_tol=tol_rad) or
            math.isclose(diff, -math.pi / 2, abs_tol=tol_rad) or
            math.isclose(diff, -3 * math.pi / 2, abs_tol=tol_rad))

def is_rectangle(points, img_width, img_height, tolerance_factor=1e-2):
    """
    Check if four points form a rectangle within a given tolerance, adjusted for image dimensions.

    Parameters:
    points (list of tuple): List of four tuples representing the points (x, y).
    img_width (int): Width of the image.
    img_height (int): Height of the image.
    tolerance_factor (float): Factor to determine tolerance as a fraction of the image diagonal.
    
    Returns:
    bool: True if the points form a rectangle, False otherwise.
    """
    if len(points) != 4:
        return False

    # Calculate the diagonal length of the image
    image_diagonal = np.sqrt(img_width**2 + img_height**2)

    # Calculate tolerance based on the image diagonal
    tolerance = tolerance_factor * image_diagonal

    # Convert points to numpy arrays for vector operations
    points = [np.array(p) for p in points]

    # Calculate the distances between all pairs of points
    dists = []
    for (p1, p2) in combinations(points, 2):
        dists.append((np.linalg.norm(p1 - p2), (p1, p2)))

    # Sort distances for easier comparison
    dists.sort(key=lambda x: x[0])

    # Check pairs of distances
    side1 = dists[0][0]
    side2 = dists[1][0]
    side3 = dists[2][0]
    side4 = dists[3][0]
    diag1 = dists[4][0]
    diag2 = dists[5][0]

    # Check if we have two pairs of equal sides and two equal diagonals
    return (abs(side1 - side2) < tolerance and
            abs(side3 - side4) < tolerance and
            abs(diag1 - diag2) < tolerance)

def order_points(pts):
    """
    Order points in the following order: top-left, bottom-left, bottom-right, top-right.
    
    Parameters:
    pts (array): Array of four points (x, y).

    Returns:
    array: Ordered array of points.
    """
    rect = np.zeros((4, 2), dtype="float32")  # Initialize an empty array for ordered points 
    s = pts.sum(axis=1)  # Sum of x and y coordinates for each point
    diff = np.diff(pts, axis=1)  # Difference between x and y coordinates for each point 
    rect[0] = pts[np.argmin(s)]  # Top-left point has the smallest sum
    rect[2] = pts[np.argmax(s)]  # Bottom-right point has the largest sum
    rect[1] = pts[np.argmin(diff)]  # Bottom-left point has the smallest difference
    rect[3] = pts[np.argmax(diff)]  # Top-right point has the largest difference
    
    return rect  # Return the ordered points

def find_rectangles(points,img_width, img_height, tolerance_factor = 0.005):
    rectangles = []
    for combination in combinations(points, 4):
        if is_rectangle(combination,img_width = img_width,img_height = img_height, tolerance_factor=tolerance_factor):
            rectangles.append(combination)
    return np.array(rectangles)

def calculate_angle(points):
    """
    Calculate the angle of rotation of a rectangle given its four corner points.

    Parameters:
    points (list of tuple): List of four points (x, y) defining the rectangle.

    Returns:
    float: The angle of rotation in degrees.
    """
    # Order the points
    ordered_points = order_points(np.array(points))
    # Extract the top-left and top-right points
    (tl, bl, br, tr) = ordered_points
    # Calculate the angle with respect to the horizontal axis
    di = tr[0] - tl[0]
    dj = tr[1] - tl[1]
    angle_rad = np.arctan2(di, dj)
    angle_deg = np.degrees(angle_rad)

    return angle_deg

def rotate_points(points, angle, center, rotation_center):
    """
    Rotate points around a center by a given angle.

    Parameters:
    points (array): Array of points (x, y) to rotate.
    angle (float): The rotation angle in degrees.
    center (tuple): The center of rotation (x, y).

    Returns:
    array: Rotated points.
    """
    # Convert the angle from degrees to radians
    angle_rad = np.radians(angle)
    # Calculate the rotation matrix
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)
    rotation_matrix = np.array([
        [cos_theta, -sin_theta],
        [sin_theta, cos_theta]
    ])
    # Translate points to origin (relative to center)
    translated_points = points - center
    # Apply the rotation matrix
    rotated_translated_points = np.dot(translated_points, rotation_matrix.T)
    # Translate points back to the original center
    rotated_points = rotated_translated_points + rotation_center
    
    return rotated_points



def remove_close_points(points, eps=20, min_samples=1):
    """
    Removes points that are too close to each other using DBSCAN clustering.

    Parameters:
    points (np.ndarray): Array of points to filter.
    eps (float): The maximum distance between two samples for them to be considered as in the same neighborhood.
    min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.

    Returns:
    np.ndarray: Array of filtered points.
    """
    if len(points) == 0:
        return np.array([])
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = db.labels_
    unique_labels = set(labels)
    filtered_points = []

    for label in unique_labels:
        if label == -1:
            continue  # Ignore noise points
        class_member_mask = (labels == label)
        cluster_points = points[class_member_mask]
        centroid = cluster_points.mean(axis=0)
        centroid = np.floor(centroid).astype(int)  # Use centroid of the cluster as the representative point
        filtered_points.append(centroid)
    return np.array(filtered_points)

def extract_quadrilateral_region(image, points):
    # Convert points to integer coordinates
    points_np = np.array(points, dtype=np.int32)
    i, j, h, w = cv2.boundingRect(points_np)
    print((i, j, h, w))  
    # Crop the extracted region to the bounding box
    cropped_region = image[i:i+h, j:j+w]

    return cropped_region

#######################################################################################################
#######################################################################################################

if __name__ == "__main__":
    
    ########### Load and preprocess the image ###############
    img_path = 'Assignment 2/im5.jpg'
    #########################################################
    
    img = Image.open(fp=img_path)
    # img = img.resize((510, 660))
    # Extract filename and extension for saving cropped images
    script_directory = os.path.dirname(os.path.abspath(__file__))
    base_filename = os.path.splitext(os.path.basename(img_path))[0]
    base_extension = os.path.splitext(img_path)[1]
    save_directory = "Assignment 2"  # Change to the desired save directory
    # Convert the image to grayscale
    img_grayscale = img.convert("L")
    img_grayscale = np.array(img_grayscale)
    # Perform edge detection using Canny edge detector from skimage
    img_canny = feature.canny(img_grayscale, sigma=7, low_threshold=1, high_threshold=20)  # Adjusted parameters
    # Perform Harris corner detection and find corner peaks
    R = my_corner_harris(img_grayscale / 255.0, k=0.05, sigma=3)
    corners = my_corner_peaks(R, rel_threshold=0.005)
    # Parameters for Hough Transform
    d_rho = 1
    d_theta = np.pi / 360
    n = 40

    # Perform Hough Transform
    H, L, res = my_hough_transform(img_canny, d_rho, d_theta, n)
    # Convert detected lines from polar to Cartesian
    lines = convert_lines_polar_to_cartesian(L)
    intersections = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):  # Ensure j > i to avoid duplicate checks
            # Check if the angle difference or its complement is within the specified bounds (tol_deg)
            if are_perpendicular(L[i][1], L[j][1],tol_deg = 3):
                temp = find_intersection(lines[i], lines[j])
                if temp is not None:
                    # Floor the intersection point and convert to integers
                    temp = np.floor(temp).astype(int)
                    intersections.append(temp)
            
    H,W = np.shape(img_grayscale)
    normalized_intersections = []
    for point in intersections:
        x, y = point
        # Scale x and y to fit within image dimensions
        x_norm = np.clip(x, 0, W - 1)
        y_norm = np.clip(y, 0, H - 1)
        normalized_intersections.append((x_norm, y_norm))
    normalized_intersections = np.array(normalized_intersections)

    # List to store filtered intersections
    filtered_intersections = []
    distance_threshold = min(H, W) * 0.005
    # Calculate distance and filter intersections
    for intersection in normalized_intersections:
        for corner in corners:
            distance = np.sqrt((intersection[0] - corner[1])**2 + (intersection[1] - corner[0])**2)
            if distance < distance_threshold:  # Adjust the distance threshold if necessary
                filtered_intersections.append(corner.astype(int))

    filtered_intersections = np.array(filtered_intersections).astype(int)
    # Sort points for consistent order
    filtered_intersections = filtered_intersections[np.lexsort((filtered_intersections[:, 1], filtered_intersections[:, 0]))]
    # Remove close points from filtered_intersections
    unique_filtered_intersections = remove_close_points(filtered_intersections, eps=20)
    print("Filtered Intersections (after removal):", unique_filtered_intersections)

    # plt.figure(figsize=(10, 8))
    # plt.imshow(img, cmap='gray')
    # for point in unique_filtered_intersections:
    #     plt.plot(point[1], point[0], 'bo')  # 'bo' for blue circles
    # plt.title('Filtered Intersections')
    # plt.figure(figsize=(10, 8))
    # plt.imshow(img, cmap='gray')
    # for point in corners:
    #     plt.plot(point[1], point[0], 'ro')  # 'ro' for red circles
    # plt.title('Corners')
    # plt.show()

    img_array = np.array(img)
    rectangles = find_rectangles(unique_filtered_intersections,W,H,tolerance_factor=0.006)
    print(f"Number of individual images found: {len(rectangles)}")

    for idx, rect in enumerate(rectangles, start=1):
        img_grayscale = np.array(img)
        # plt.imshow(img_grayscale, cmap='gray')
        # plt.title('Harris with Detected Corners')
        # plt.scatter(rect[:, 1], rect[:, 0], color='red', marker='s', s=1)
        # plt.show()
        sorted_points = order_points(rect)
        angle = calculate_angle(sorted_points)
        if abs(angle) > 0.5:
            height, width = img_grayscale.shape[:2]
            img_grayscale = my_img_rotation(img_grayscale, angle*(np.pi/180))
            # plt.imshow(img_grayscale, cmap='gray')
            # plt.title('Rotated Image')
            # plt.show()
            new_height, new_width = img_grayscale.shape[:2]
            center_i = height // 2
            center_j = width // 2 
            rot_center_i = new_height / 2 
            rot_center_j = new_width / 2
            rect = rotate_points(rect, angle, center=(center_i,center_j), rotation_center=(rot_center_i, rot_center_j))
        
        region = extract_quadrilateral_region(img_grayscale, rect)
        # plt.imshow(region, cmap='gray')
        # plt.title('Region')
        # plt.show()
        # Save the cropped image
        cropped_img = Image.fromarray(region)
        cropped_img_path = os.path.join(script_directory, f"{base_filename}_{idx}{base_extension}")
        cropped_img.save(cropped_img_path)
        print(f"Saved cropped image: {cropped_img_path}")
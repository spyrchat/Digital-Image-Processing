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
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def find_rectangles(points, img_width, img_height, tolerance_factor=0.005):
    rectangles = []
    for combination in combinations(points, 4):
        if is_rectangle(combination, img_width=img_width, img_height=img_height, tolerance_factor=tolerance_factor):
            rectangles.append(combination)
    return np.array(rectangles)

def calculate_angle(points):
    ordered_points = order_points(np.array(points))
    (tl, bl, br, tr) = ordered_points
    di = tr[0] - tl[0]
    dj = tr[1] - tl[1]
    angle_rad = np.arctan2(di, dj)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def rotate_points(points, angle, center, rotation_center):
    angle_rad = np.radians(angle)
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)
    rotation_matrix = np.array([
        [cos_theta, -sin_theta],
        [sin_theta, cos_theta]
    ])
    translated_points = points - center
    rotated_translated_points = np.dot(translated_points, rotation_matrix.T)
    rotated_points = rotated_translated_points + rotation_center
    
    return rotated_points

def remove_close_points(points, eps=20, min_samples=1):
    if len(points) == 0:
        return np.array([])
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = db.labels_
    unique_labels = set(labels)
    filtered_points = []

    for label in unique_labels:
        if label == -1:
            continue
        class_member_mask = (labels == label)
        cluster_points = points[class_member_mask]
        centroid = cluster_points.mean(axis=0)
        centroid = np.floor(centroid).astype(int)
        filtered_points.append(centroid)
    return np.array(filtered_points)

def extract_quadrilateral_region(image, points):
    points_np = np.array(points, dtype=np.int32)
    i, j, h, w = cv2.boundingRect(points_np)
    cropped_region = image[i:i+h, j:j+w]
    return cropped_region

if __name__ == "__main__":
    ########### Load and preprocess the image ###############
    img_path = 'Assignment 2/im1.jpg'
    #########################################################

    img = Image.open(fp=img_path)
    img_high_res = img.convert("L")
    img_high_res = np.array(img_high_res)

    # Resize the image for lower resolution processing
    scale_factor = 0.2
    img_low_res = cv2.resize(img_high_res, (int(img_high_res.shape[1] * scale_factor), int(img_high_res.shape[0] * scale_factor)))

    # Extract filename and extension for saving cropped images
    script_directory = os.path.dirname(os.path.abspath(__file__))
    base_filename = os.path.splitext(os.path.basename(img_path))[0]
    base_extension = os.path.splitext(img_path)[1]
    # save_directory = "Assignment 2"

    # Perform edge detection using Canny edge detector from skimage
    img_canny = feature.canny(img_low_res, sigma=2.5, low_threshold=10, high_threshold=50)

    # Perform Harris corner detection and find corner peaks
    R = my_corner_harris(img_low_res / 255.0, k=0.05, sigma=3)
    corners = my_corner_peaks(R, rel_threshold=0.005)

    # Rescale corners to high resolution
    rescaled_corners = (corners / scale_factor).astype(int)

    # Parameters for Hough Transform
    d_rho = 1
    d_theta = np.pi / 360
    n = 30

    # Perform Hough Transform
    H, L, res = my_hough_transform(img_canny, d_rho, d_theta, n)
    # Convert detected lines from polar to Cartesian
    lines = convert_lines_polar_to_cartesian(L)
    intersections = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            if are_perpendicular(L[i][1], L[j][1], tol_deg=3):
                temp = find_intersection(lines[i], lines[j])
                if temp is not None:
                    temp = np.floor(temp).astype(int)
                    intersections.append(temp)

    H, W = img_low_res.shape
    normalized_intersections = []
    for point in intersections:
        x, y = point
        x_norm = np.clip(x, 0, W - 1)
        y_norm = np.clip(y, 0, H - 1)
        normalized_intersections.append((x_norm, y_norm))
    normalized_intersections = np.array(normalized_intersections)

    filtered_intersections = []
    distance_threshold = min(H, W) * 0.005
    for intersection in normalized_intersections:
        for corner in corners:
            distance = np.sqrt((intersection[0] - corner[1])**2 + (intersection[1] - corner[0])**2)
            if distance < distance_threshold:
                filtered_intersections.append(corner.astype(int))

    filtered_intersections = np.array(filtered_intersections).astype(int)
    filtered_intersections = filtered_intersections[np.lexsort((filtered_intersections[:, 1], filtered_intersections[:, 0]))]
    unique_filtered_intersections = remove_close_points(filtered_intersections, eps=20)
    print("Filtered Intersections (after removal):", unique_filtered_intersections)

    # Rescale the filtered intersections to match the original high-resolution image dimensions
    rescaled_intersections = (unique_filtered_intersections / scale_factor).astype(int)
    print("Rescaled Intersections (after removal):", rescaled_intersections)

    img_array = np.array(img)
    rectangles = find_rectangles(rescaled_intersections, img_high_res.shape[1], img_high_res.shape[0], tolerance_factor=0.006)
    print(f"Number of individual images found: {len(rectangles)}")

    for idx, rect in enumerate(rectangles, start=1):
        img_grayscale = np.array(img)
        sorted_points = order_points(rect)
        angle = calculate_angle(sorted_points)
        if abs(angle) > 0.5:
            height, width = img_grayscale.shape[:2]
            img_grayscale = my_img_rotation(img_grayscale, angle * (np.pi / 180))
            new_height, new_width = img_grayscale.shape[:2]
            center_i = height // 2
            center_j = width // 2
            rot_center_i = new_height / 2
            rot_center_j = new_width / 2
            rect = rotate_points(rect, angle, center=(center_i, center_j), rotation_center=(rot_center_i, rot_center_j))

        region = extract_quadrilateral_region(img_grayscale, rect)
        cropped_img = Image.fromarray(region)
        cropped_img_path = os.path.join(script_directory, f"{base_filename}_{idx}{base_extension}")
        cropped_img.save(cropped_img_path)
        print(f"Saved cropped image: {cropped_img_path}")

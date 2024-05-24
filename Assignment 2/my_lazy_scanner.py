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
    """
    Convert polar coordinates (rho, theta) to Cartesian line parameters (A, B, C).

    Parameters:
    rho (float): Distance from the origin to the line.
    theta (float): Angle between the normal to the line and the x-axis.

    Returns:
    tuple: Coefficients (A, B, C) of the line equation Ax + By + C = 0.
    """
    A = np.cos(theta)
    B = np.sin(theta)
    C = -rho
    return A, B, C

def convert_lines_polar_to_cartesian(lines):
    """
    Convert a list of lines from polar coordinates to Cartesian coordinates.

    Parameters:
    lines (list of tuples): List of lines in polar coordinates (rho, theta).

    Returns:
    list of tuples: List of lines in Cartesian coordinates (A, B, C).
    """
    cartesian_lines = []
    for rho, theta in lines:
        A, B, C = polar_to_cartesian(rho, theta)
        cartesian_lines.append((A, B, C))
    return cartesian_lines

def find_intersection(line1, line2):
    """
    Find the intersection point of two lines (Ax + By + C = 0) given in Cartesian coordinates.

    Parameters:
    line1 (tuple): Coefficients (A1, B1, C1) of the first line.
    line2 (tuple): Coefficients (A2, B2, C2) of the second line.

    Returns:
    np.ndarray: Intersection point (x, y) or None if lines are parallel.
    """
    A1, B1, C1 = line1
    A2, B2, C2 = line2
    A = np.array([[A1, B1], [A2, B2]])
    B = np.array([-C1, -C2])
    if np.linalg.det(A) == 0:
        return None
    else:
        intersection_point = np.linalg.solve(A, B)
        return intersection_point

def are_perpendicular(theta1, theta2, tol_deg=3):
    """
    Check if two angles are perpendicular within a given tolerance.

    Parameters:
    theta1 (float): First angle in radians.
    theta2 (float): Second angle in radians.
    tol_deg (float): Tolerance in degrees.

    Returns:
    bool: True if the angles are perpendicular within the tolerance, False otherwise.
    """
    tol_rad = math.radians(tol_deg)
    theta1 = theta1 % (2 * math.pi)
    theta2 = theta2 % (2 * math.pi)
    diff = abs(theta1 - theta2)
    return (math.isclose(diff, math.pi / 2, abs_tol=tol_rad) or 
            math.isclose(diff, 3 * math.pi / 2, abs_tol=tol_rad) or
            math.isclose(diff, -math.pi / 2, abs_tol=tol_rad) or
            math.isclose(diff, -3 * math.pi / 2, abs_tol=tol_rad))

def is_rectangle(points, img_width, img_height, tolerance_factor=1e-2):
    """
    Check if four points form a rectangle within a given tolerance.

    Parameters:
    points (list of tuples): List of four points (x, y).
    img_width (int): Width of the image.
    img_height (int): Height of the image.
    tolerance_factor (float): Tolerance factor for comparing distances.

    Returns:
    bool: True if the points form a rectangle, False otherwise.
    """
    if len(points) != 4:
        return False

    image_diagonal = np.sqrt(img_width**2 + img_height**2)
    tolerance = tolerance_factor * image_diagonal
    points = [np.array(p) for p in points]
    dists = []
    for (p1, p2) in combinations(points, 2):
        dists.append((np.linalg.norm(p1 - p2), (p1, p2)))
    dists.sort(key=lambda x: x[0])

    side1 = dists[0][0]
    side2 = dists[1][0]
    side3 = dists[2][0]
    side4 = dists[3][0]
    diag1 = dists[4][0]
    diag2 = dists[5][0]

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
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def find_rectangles(points, img_width, img_height, tolerance_factor=0.005):
    """
    Find rectangles among a set of points.

    Parameters:
    points (array): Array of points (x, y).
    img_width (int): Width of the image.
    img_height (int): Height of the image.
    tolerance_factor (float): Tolerance factor for comparing distances.

    Returns:
    np.ndarray: Array of rectangles (each rectangle is an array of four points).
    """
    rectangles = []
    for combination in combinations(points, 4):
        if is_rectangle(combination, img_width=img_width, img_height=img_height, tolerance_factor=tolerance_factor):
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
    ordered_points = order_points(np.array(points))
    (tl, bl, br, tr) = ordered_points
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
    center (tuple): The center of the original image (x, y).
    rotation_center (tuple): The center of the rotated image (x, y).

    Returns:
    array: Rotated points.
    """
    angle_rad = np.radians(angle)
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)
    rotation_matrix = np.array([[cos_theta, -sin_theta],[sin_theta, cos_theta]])
    translated_points = points - center
    rotated_translated_points = np.dot(translated_points, rotation_matrix.T)
    rotated_points = rotated_translated_points + rotation_center
    return rotated_points

def remove_close_points(points, eps=20, min_samples=1):
    """
    Remove points that are too close to each other using DBSCAN clustering.

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
            continue
        class_member_mask = (labels == label)
        cluster_points = points[class_member_mask]
        centroid = cluster_points.mean(axis=0)
        centroid = np.floor(centroid).astype(int)
        filtered_points.append(centroid)
    return np.array(filtered_points)

def extract_rectangular_region(image, points):
    """
    Extract a rectangular region from the image given four corner points.

    Parameters:
    image (np.ndarray): The original image.
    points (array): Array of four corner points (x, y).

    Returns:
    np.ndarray: The cropped rectangular region.
    """
    points_np = np.array(points, dtype=np.int32)
    i, j, h, w = cv2.boundingRect(points_np)
    cropped_region = image[i:i+h, j:j+w]
    return cropped_region

if __name__ == "__main__":
    ########### Load and preprocess the image ###############
    img_path = 'Assignment 2/im4.jpg'
    #########################################################

    img = Image.open(fp=img_path)
    img_high_res = img.convert("L")
    img_high_res = np.array(img_high_res)

    # Resize the image for lower resolution processing
    scale_factor = 0.1
    img_low_res = cv2.resize(img_high_res, (int(img_high_res.shape[1] * scale_factor), int(img_high_res.shape[0] * scale_factor)))

    # Extract filename and extension for saving cropped images
    script_directory = os.path.dirname(os.path.abspath(__file__))
    base_filename = os.path.splitext(os.path.basename(img_path))[0]
    base_extension = os.path.splitext(img_path)[1] 

    # Perform edge detection using Canny edge detector from skimage
    img_canny = feature.canny(img_low_res, sigma=2.5, low_threshold=10, high_threshold=50)

    # Perform Harris corner detection and find corner peaks
    R = my_corner_harris(img_low_res / 255.0, sigma=3, k=0.05)
    corners = my_corner_peaks(R, rel_threshold=0.005)

    # Parameters for Hough Transform
    d_rho = 1
    d_theta = np.pi / 360
    n = 25

    # Perform Hough Transform
    H, L, res = my_hough_transform(img_canny, d_rho, d_theta, n)
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

# find the intersections that are close (within distance_tolorance) to the corners that were found from Harris Algorithm
    filtered_intersections = []
    distance_threshold = max(H, W) * 0.005
    for intersection in normalized_intersections:
        for corner in corners:
            distance = np.sqrt((intersection[0] - corner[1])**2 + (intersection[1] - corner[0])**2)
            if distance < distance_threshold:
                filtered_intersections.append(corner.astype(int))

    filtered_intersections = np.array(filtered_intersections).astype(int)
    filtered_intersections = filtered_intersections[np.lexsort((filtered_intersections[:, 1], filtered_intersections[:, 0]))]
    unique_filtered_intersections = remove_close_points(filtered_intersections, eps=20)
#Find the Rectangles
    rectangles = find_rectangles(unique_filtered_intersections, img_low_res.shape[1], img_low_res.shape[0], tolerance_factor=0.006)
    print(f"Number of individual images found: {len(rectangles)}")

    # Rescale rectangles to the high-resolution image dimensions
    rescaled_rectangles = []
    for rect in rectangles:
        rescaled_rect = (np.array(rect) / scale_factor).astype(int)
        rescaled_rectangles.append(rescaled_rect)

    for idx, rect in enumerate(rescaled_rectangles, start=1):
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

        region = extract_rectangular_region(img_grayscale, rect)
        cropped_img = Image.fromarray(region)
        cropped_img_path = os.path.join(script_directory, f"{base_filename}_{idx}{base_extension}")
        cropped_img.save(cropped_img_path)
        print(f"Saved cropped image: {cropped_img_path}")

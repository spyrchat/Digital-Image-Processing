import numpy as np
from global_hist_eq import get_equalization_transform_of_img
from PIL import Image

def calculate_eq_transformations_of_regions(img_array: np.ndarray,
region_len_h: int,region_len_w: int):
    W = img_array.shape[1]
    H = img_array.shape[0]
    num_blocks_w = W// region_len_w + (1 if W % region_len_w > 0 else 0)
    num_blocks_h = H // region_len_h + (1 if H % region_len_h > 0 else 0)
    total_blocks = num_blocks_w * num_blocks_h
    top_left_corners = []
    region_to_eq_transform = {}

    for j in range(num_blocks_h):
        for i in range(num_blocks_w):
            top_left_x = i * region_len_w
            top_left_y = j * region_len_h

            bottom_right_x = min((i+1) * region_len_w, W)
            bottom_right_y = min((j+1) * region_len_h, H)

            region = img_array[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
            region_to_eq_transform[(top_left_y,top_left_x)] = get_equalization_transform_of_img(region) 
    # print(region_to_eq_transform.keys())
    # print(region_to_eq_transform.__len__())
    return region_to_eq_transform


def find_four_closest_centers(points, known_point):
    
    points_array = np.array(points)
    diff = points_array - known_point
    squared_diff = diff ** 2
    squared_distances = np.sum(squared_diff, axis=1)
    closest_indices = np.argsort(squared_distances)[:4]
    closest_centers = [points[i] for i in closest_indices]
    return closest_centers
    
def perform_adaptive_hist_equalization(img_array: np.ndarray,region_len_h: int,region_len_w: int):
    W = img_array.shape[1]
    H = img_array.shape[0]
    transformation_dict = calculate_eq_transformations_of_regions(img_array,region_len_h,region_len_w)
    equalized_img = np.zeros_like(img_array)
    num_blocks_w = W// region_len_w + (1 if W % region_len_w > 0 else 0)
    num_blocks_h = H // region_len_h + (1 if H % region_len_h > 0 else 0)
    top_left_corners = []
    for i in range(0, H, region_len_h):
        for j in range(0, W, region_len_w):
             top_left_corners.append((i, j))
    centers = []
    for i in top_left_corners:
        center_y = i[0] + region_len_h // 2
        center_x = i[1] + region_len_w // 2
        centers.append((center_y, center_x))
    # print("centers",centers)
    # print(len(centers))
    
    for y in range(H):
        for x in range(W):
            block_j = y // region_len_h
            block_i = x // region_len_w

            if (y < region_len_h // 2) or (x < region_len_w // 2) or (y >= H - region_len_h // 2) or (x >= W - region_len_w // 2):
                # Pixel is in the outer region of a block
                pixel_value = img_array[y, x]
                transform = transformation_dict[(block_j * region_len_h, block_i * region_len_w)]
                equalized_img[y, x] = transform[pixel_value]
            else:
                block_center_y = block_j * region_len_h + region_len_h // 2
                block_center_x = block_i * region_len_w + region_len_w // 2
                contextual_neighbours = find_four_closest_centers(centers, (y, x))
                w_left, w_right = None, None
                h_up, h_down = None, None

                for i in contextual_neighbours:
                    if i[1] < x and (w_left is None or i[1] > w_left):
                        w_left = i[1]
                    if i[1] >= x and (w_right is None or i[1] < w_right):
                        w_right = i[1]
                    if i[0] > y and (h_down is None or i[0] < h_down):
                        h_down = i[0]
                    if i[0] <= y and (h_up is None or i[0] > h_up):
                        h_up = i[0]
                
                if w_left is None or w_right is None or h_up is None or h_down is None:
                    transform = transformation_dict[(block_j * region_len_h, block_i * region_len_w)]
                    equalized_img[y, x] = transform[pixel_value]
                    # raise ValueError("Could not find four closest centers for pixel ({}, {})".format(y, x))
                
                else:
                # Calculate interpolation weights
                    a = (x - w_left) / (w_right - w_left) if w_right != w_left else 0.5
                    b = (y - h_down) / (h_up - h_down) if h_up != h_down else 0.5

                    # Interpolate the pixel value using the transformation functions
                    T_tl = transformation_dict.get((h_down - (region_len_h // 2), w_left - (region_len_w //2)), np.zeros(256))
                    T_tr = transformation_dict.get((h_down - (region_len_h // 2), w_right - (region_len_w // 2)), np.zeros(256))
                    T_bl = transformation_dict.get((h_up - (region_len_h // 2), w_left - (region_len_w // 2)), np.zeros(256))
                    T_br = transformation_dict.get((h_up -  (region_len_h // 2), w_right - (region_len_w // 2)), np.zeros(256))
                    
                    pixel_value = img_array[y, x]
                    interpolated_value = (
                        a * b * T_br[pixel_value] +
                        a * (1 - b) * T_tr[pixel_value] +
                        (1 - a) * b * T_bl[pixel_value] +
                        (1 - a) * (1 - b) * T_tl[pixel_value]
                    )
                    equalized_img[y, x] = np.round(interpolated_value)

    return equalized_img


calculate_eq_transformations_of_regions(np.array(Image.open("Assignment 1/input_img.png").convert("L")), 48, 64)






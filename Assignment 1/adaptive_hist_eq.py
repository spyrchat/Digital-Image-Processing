import numpy as np
from global_hist_eq import get_equalization_transform_of_img
from PIL import Image

def calculate_eq_transformations_of_regions(img_array: np.ndarray, region_len_h: int,region_len_w: int):
    W = img_array.shape[1]
    H = img_array.shape[0]
    # Calculate the number of blocks in the horizontal and vertical directions,
    # taking into account the possibility that the image dimensions are not divisible by the region length
    num_blocks_w = W// region_len_w + (1 if W % region_len_w > 0 else 0)
    num_blocks_h = H // region_len_h + (1 if H % region_len_h > 0 else 0)
    region_to_eq_transform = {}

    for j in range(num_blocks_h):
        for i in range(num_blocks_w):
            top_left_x = i * region_len_w
            top_left_y = j * region_len_h
            # The min function is used to ensure that the bottom right corner of the region does not exceed the image dimensions
            bottom_right_x = min((i+1) * region_len_w, W)
            bottom_right_y = min((j+1) * region_len_h, H)
            # The top left corner of the region is used as the key for the transformation dictionary
            region = img_array[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
            region_to_eq_transform[(top_left_y,top_left_x)] = get_equalization_transform_of_img(region) 

    return region_to_eq_transform



def perform_adaptive_hist_equalization(img_array: np.ndarray,region_len_h: int,region_len_w: int):
    W = img_array.shape[1]
    H = img_array.shape[0]
    transformation_dict = calculate_eq_transformations_of_regions(img_array,region_len_h,region_len_w)
    equalized_img = np.zeros_like(img_array)
    top_left_corners = []
    
    for i in range(0, H, region_len_h):
        for j in range(0, W, region_len_w):
             top_left_corners.append((i, j))
    centers = []

    for i in top_left_corners:

        center_y = i[0] + region_len_h // 2
        center_x = i[1] + region_len_w // 2
        centers.append((center_y, center_x))

    for y in range(H):
        for x in range(W):
            block_j = y // region_len_h
            block_i = x // region_len_w

            #Pixel is in the outer region, use the transformation of the region
            if (y <= region_len_h // 2) or (x <= region_len_w // 2) or (y >= H - region_len_h // 2) or (x >= W - region_len_w // 2):
                pixel_value = img_array[y, x]
                transform = transformation_dict[(block_j * region_len_h, block_i * region_len_w)]
                equalized_img[y, x] = transform[pixel_value]
            else:
                # Pixel is a center itself, use the transformation of the block it belongs to
                if (y,x) in centers:
                    transform = transformation_dict[(block_j * region_len_h, block_i * region_len_w)]
                    equalized_img[y, x] = transform[img_array[y, x]]
                
                else:
                    w_left, w_right = None, None
                    h_up, h_down = None, None
                    
                    w_left, w_right, h_up, h_down = find_w_and_h(centers, (y, x), region_len_h, region_len_w)
                    
                    if w_left is None or w_right is None or h_up is None or h_down is None:
                        raise ValueError("Could not find four closest centers for pixel ({}, {})".format(y, x))
                    
                    else:
                        # Calculate interpolation weights a and b for the pixel and check for division by zero
                        # The division by zero is handled by setting the value to 0.01. OF COURSE the way the code works
                        # is that it will never divide by zero, because w_right and w_left or h_up and h_down cant be the same
                        # but it is good to have a check in place. Or maybe I am just paranoid...
                        a = (x - w_left) / (w_right - w_left) if w_right != w_left else 0.01
                        b = (y - h_down) / (h_up - h_down) if h_up != h_down else 0.01

                        # Interpolate the pixel value using the transformation functions
                        # The centers are tranlated to the top left corner of the region 
                        # Because the transformation functions are stored with respect to the top left corner of the region
                        T_tl = transformation_dict.get((h_down - (region_len_h // 2), w_left - (region_len_w // 2)), np.zeros(256))
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
                        equalized_img[y, x] = interpolated_value

    return equalized_img


def perform_adaptive_hist_equalization_no_interpolation(img_array: np.ndarray,region_len_h: int,region_len_w: int):
    W = img_array.shape[1]
    H = img_array.shape[0]
    transformation_dict = calculate_eq_transformations_of_regions(img_array,region_len_h,region_len_w)
    equalized_img = np.zeros_like(img_array)
    top_left_corners = []
    
    for i in range(0, H, region_len_h):
        for j in range(0, W, region_len_w):
             top_left_corners.append((i, j))
    centers = []

    for i in top_left_corners:
        center_y = i[0] + region_len_h // 2
        center_x = i[1] + region_len_w // 2
        centers.append((center_y, center_x))

    for y in range(H):
        for x in range(W):
            block_j = y // region_len_h
            block_i = x // region_len_w

        
            pixel_value = img_array[y, x]
            transform = transformation_dict[(block_j * region_len_h, block_i * region_len_w)]
            equalized_img[y, x] = transform[pixel_value]
           

    return equalized_img

#================================================================================================#
# The function below is used to find the closest center to a given pixel. Afer finding the closest
# center, the function figures out the quadrants in which the pixel lies with the nearest center as
# the origin and returns the w_left, w_right, h_up, h_down values for the pixel. It uses numpy arrays
# for calculation of the nearest center in order to optimize the code.
#================================================================================================#

def find_w_and_h(points, pixel, region_len_h, region_len_w):
    points_array = np.array(points)
    diff = points_array - pixel
    squared_diff = diff ** 2
    squared_distances = np.sum(squared_diff, axis=1)
    closest_indices = np.argsort(squared_distances)[:1]
    closest_center =  points_array[closest_indices]

    #Translate the known point to the coordinate system with the closest center as the origin
    delta_x = pixel[1] - closest_center[0][1]
    delta_y = pixel[0] - closest_center[0][0]
    
    # The idea is that if I know the closest center to a pixel, I can figure out the quadrant in which the pixel lies with respect to the closest center.
    # After that I can calculate the w_left, w_right, h_up, h_down values for the pixel accordingly, since there is symmetry between the 4 centers in a region.
    
    # 1st quadrant, here points that lie on the y-axis are also considered to be in the 1st quadrant
    if delta_x > 0 and delta_y <= 0:
        w_left = closest_center[0][1]
        w_right = closest_center[0][1] + region_len_w
        h_up = closest_center[0][0] - region_len_h
        h_down = closest_center[0][0]
    # 2nd quadrant here points that lie on the x-axis, are also considered to be in the 2nd quadrant
    elif delta_x <= 0 and delta_y < 0:
        w_left = closest_center[0][1] - region_len_w
        w_right = closest_center[0][1]
        h_up = closest_center[0][0] - region_len_h
        h_down = closest_center[0][0]
    # 3rd quadrant, here points that lie on the y-axis are also considered to be in the 3rd quadrant
    elif delta_x < 0 and delta_y >= 0:
        w_left = closest_center[0][1] - region_len_w
        w_right = closest_center[0][1]
        h_up = closest_center[0][0]
        h_down = closest_center[0][0] + region_len_h
    # 4th quadrant, here points that lie on the x-axis are also considered to be in the 4th quadrant  
    elif delta_x >= 0 and delta_y > 0:
        w_left = closest_center[0][1]
        w_right = closest_center[0][1] + region_len_w
        h_up = closest_center[0][0]
        h_down = closest_center[0][0] + region_len_h
    
    return w_left, w_right, h_up, h_down
import numpy as np
from global_hist_eq import get_equalization_transform_of_img

def calculate_eq_transformations_of_regions(img_array: np.ndarray,
region_len_h: int,region_len_w: int):
    W = img_array.shape[1]
    H = img_array.shape[0]
    num_blocks_w = W// region_len_w + (1 if W % region_len_w > 0 else 0)
    num_blocks_h = H // region_len_h + (1 if H % region_len_h > 0 else 0)
    total_blocks = num_blocks_w * num_blocks_h
    top_left_corners = []
    region_to_eq_transform = {}
    # # The outer loop should iterate over the height
    # for i in range(0, H, region_len_h):
    #     # The inner loop should iterate over the width
    #     for j in range(0, W, region_len_w):
    #         top_left_corners.append((i, j))

    # print(top_left_corners)
    # assert len(top_left_corners) == total_blocks

    for j in range(num_blocks_h):
        for i in range(num_blocks_w):
            top_left_x = i * region_len_w
            top_left_y = j * region_len_h

            bottom_right_x = min((i+1) * region_len_w, W)
            bottom_right_y = min((j+1) * region_len_h, H)

            region = img_array[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
            region_to_eq_transform[(top_left_y,top_left_x)] = get_equalization_transform_of_img(region) 

    return region_to_eq_transform

    

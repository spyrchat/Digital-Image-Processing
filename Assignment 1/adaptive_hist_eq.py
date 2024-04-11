import numpy as np
from global_hist_eq import get_equalization_transform_of_img

def calculate_eq_transformations_of_regions(img_array: np.ndarray,
region_len_h: int,region_len_w: int):
    W = img_array.shape[1]
    H = img_array.shape[0]
    num_blocks_w = W// region_len_w + (1 if W % region_len_w > 0 else 0)
    num_blocks_h = H // region_len_h + (1 if H % region_len_h > 0 else 0)

    top_left_corners = []
    for i in range(0, W, region_len_w):
        for j in range(0, H, region_len_h):
            top_left_corners.append((i, j))

    print(top_left_corners)
    assert len(top_left_corners) == num_blocks_w * num_blocks_h


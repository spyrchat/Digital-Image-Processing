import numpy as np

def calculate_eq_transformations_of_regions(img_array: np.ndarray,
region_len_h: int,region_len_w: int):
    W = img_array.shape[1]
    H = img_array.shape[0]
    num_regions_h = H / region_len_h
    num_regions_w = W / region_len_w

    if num_regions_h % 1 != 0 or num_regions_w % 1 != 0:
        raise ValueError("Image dimensions should be divisible by region length")
    
    contexual_regions = []
    for i in range(int(num_regions_h)):
        for j in range(int(num_regions_w)):
            region = img_array[i*region_len_h:(i+1)*region_len_h,j*region_len_w:(j+1)*region_len_w]
            contexual_regions.append(region)
    


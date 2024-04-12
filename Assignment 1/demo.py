from PIL import Image
import numpy as np
from global_hist_eq import get_equalization_transform_of_img
from adaptive_hist_eq import calculate_eq_transformations_of_regions

filename = "Assignment 1/input_img.png"
img = Image.open(fp=filename)
bw_img = img.convert("L")
img_array = np.array(bw_img)

equalization_transform = get_equalization_transform_of_img(img_array)
# print(equalization_transform)
c = calculate_eq_transformations_of_regions(img_array, 48, 64)

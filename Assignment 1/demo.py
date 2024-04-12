from PIL import Image
import numpy as np
from global_hist_eq import get_equalization_transform_of_img
from adaptive_hist_eq import calculate_eq_transformations_of_regions, perform_adaptive_hist_equalization

filename = "Assignment 1/input_img.png"
img = Image.open(fp=filename)
bw_img = img.convert("L")
img_array = np.array(bw_img)

equalization_transform = get_equalization_transform_of_img(img_array)
# print(equalization_transform)
# c = calculate_eq_transformations_of_regions(img_array, 48, 64)
new_img = perform_adaptive_hist_equalization(img_array, 48, 64)
new_img = np.clip(new_img, 0, 255)  # Clip values to ensure they fall within the proper range
new_img = new_img.astype(np.uint8)  # Convert to unsigned byte format

# Convert to a PIL image
new_img_pil = Image.fromarray(new_img)

new_img.save("Assignment 1/created_image.png", "PNG")

# Display the image
new_img_pil.show()
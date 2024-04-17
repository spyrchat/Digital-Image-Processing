from PIL import Image
import numpy as np
from global_hist_eq import get_equalization_transform_of_img,perform_global_hist_equalization, plot_transformation_function, plot_histogram
from adaptive_hist_eq import calculate_eq_transformations_of_regions, perform_adaptive_hist_equalization


# Assume calculate_eq_transformations_of_regions and perform_adaptive_hist_equalization
# are defined in adaptive_hist_eq and imported correctly

# Load the image and convert it to a grayscale NumPy array
filename = "Assignment 1/input_img.png"
img = Image.open(filename)
bw_img = img.convert("L")
img_array = np.array(bw_img)


plot_transformation_function(filename)

plot_histogram(img_array)
# Perform global histogram equalization on the image array
new_img_array_global = perform_global_hist_equalization(img_array)
new_img_array_global = np.clip(new_img_array_global, 0, 255).astype(np.uint8)
new_img_pil_global = Image.fromarray(new_img_array_global) 
new_img_pil_global.save("Assignment 1/global_equalized_image.png")
new_img_pil_global.show()


# Perform adaptive histogram equalization on the image array
# with given region sizes for the local histograms
new_img_array = perform_adaptive_hist_equalization(img_array, region_len_h=48, region_len_w=64)
# Ensure the new image array is in the correct byte range and data type
new_img_array = np.clip(new_img_array, 0, 255).astype(np.uint8)
# Convert the processed image array back to a PIL image
new_img_pil = Image.fromarray(new_img_array)
# Save the new image
new_img_pil.save("Assignment 1/adaptive_equalized_image.png")
# Optionally, display the image
new_img_pil.show()
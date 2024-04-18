from PIL import Image
import numpy as np
from global_hist_eq import perform_global_hist_equalization
from adaptive_hist_eq import perform_adaptive_hist_equalization
from helper import plot_histogram, plot_transformation_function

# Assume calculate_eq_transformations_of_regions and perform_adaptive_hist_equalization
# are defined in adaptive_hist_eq and imported correctly

# Load the image and convert it to a grayscale NumPy array
filename = "Assignment 1/input_img.png"
img = Image.open(filename)
bw_img = img.convert("L")
img_array = np.array(bw_img)

# Save the original image and print
original_img = Image.fromarray(img_array)
original_img.save("Assignment 1/original_image.png")
original_img.show()
plot_histogram(img_array, "Original Histogram")

plot_transformation_function(filename)


# Perform global histogram equalization on the image array
new_img_array_global = perform_global_hist_equalization(img_array)
new_img_array_global = np.clip(new_img_array_global, 0, 255).astype(np.uint8)
new_img_pil_global = Image.fromarray(new_img_array_global) 
new_img_pil_global.save("Assignment 1/global_equalized_image.png")
new_img_pil_global.show()
plot_histogram(new_img_array_global, "Global Equalized Histogram")

new_img_array = perform_adaptive_hist_equalization(img_array, region_len_h=48, region_len_w=64)
# Ensure the new image array is in the correct byte range and data type
new_img_array = np.clip(new_img_array, 0, 255).astype(np.uint8)
new_img_pil = Image.fromarray(new_img_array)
new_img_pil.save("Assignment 1/adaptive_equalized_image.png")
new_img_pil.show()
plot_histogram(new_img_array, "Adaptive Equalized Histogram")
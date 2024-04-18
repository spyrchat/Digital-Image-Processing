from PIL import Image
import numpy as np
from global_hist_eq import perform_global_hist_equalization
from adaptive_hist_eq import perform_adaptive_hist_equalization, perform_adaptive_hist_equalization_no_interpolation
from helper import plot_histogram, plot_transformation_function, plot_histogram_side_to_side

# Load the image and convert it to a grayscale NumPy array
filename = "./input_img.png"
img = Image.open(filename)

# Resize the image to 512x384 - Optional
# The image is resized to 512x384 to fit an integer number of 48x64 regions
# The resampling method used is Lanczos and it will make the image look better when resizing
# I left this code commented out because the way the algorithms are implemented, they can work with any image size
# img.resize((512, 384), Image.Resampling.LANCZOS)

bw_img = img.convert("L")
img_array = np.array(bw_img)

# Save the original image and print
original_img = Image.fromarray(img_array)
original_img.save("./original_image.png")
original_img.show()
plot_histogram(img_array, "Original Histogram")

plot_transformation_function(filename)

plot_histogram_side_to_side(img_array)

# Perform global histogram equalization on the image array
new_img_array_global = perform_global_hist_equalization(img_array)
new_img_array_global = np.clip(new_img_array_global, 0, 255).astype(np.uint8)
new_img_pil_global = Image.fromarray(new_img_array_global) 
new_img_pil_global.save("./global_equalized_image.png")
new_img_pil_global.show()
plot_histogram(new_img_array_global, "Global Equalized Histogram")

# Perform adaptive histogram equalization on the image array
new_img_array = perform_adaptive_hist_equalization(img_array, region_len_h=48, region_len_w=64)
# Ensure the new image array is in the correct byte range and data type
new_img_array = np.clip(new_img_array, 0, 255).astype(np.uint8)
new_img_pil = Image.fromarray(new_img_array)
new_img_pil.save("./adaptive_equalized_image.png")
new_img_pil.show()
plot_histogram(new_img_array, "Adaptive Equalized Histogram")

# Perform adaptive histogram equalization without interpolation
new_img_array_no_interpolation = perform_adaptive_hist_equalization_no_interpolation(img_array, region_len_h=48, region_len_w=64)
new_img_array_no_interpolation = np.clip(new_img_array_no_interpolation, 0, 255).astype(np.uint8)
new_img_pil_no_interpolation = Image.fromarray(new_img_array_no_interpolation)
new_img_pil_no_interpolation.save("Assignment 1/adaptive_equalized_image_no_interpolation.png")
new_img_pil_no_interpolation.show()
plot_histogram(new_img_array_no_interpolation, "Adaptive Equalized Histogram_no_interpolation")
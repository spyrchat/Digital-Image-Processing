from PIL import Image
import numpy as np
from global_hist_eq import get_equalization_transform_of_img
from adaptive_hist_eq import calculate_eq_transformations_of_regions, perform_adaptive_hist_equalization


# Load the image and convert it to a grayscale NumPy array
filename = "./input_img.png"
img = Image.open(filename)
bw_img = img.convert("L")
img_array = np.array(bw_img)

# Calculate the equalization transformation for the entire image
equalization_transform = get_equalization_transform_of_img(img_array)

# Apply the histogram equalization transformation to the original image
# Remap the pixel values using the equalization transform
equalized_img_array = equalization_transform[img_array.flatten()].reshape(img_array.shape)

# Convert the equalized image array back to a PIL image
equalized_img = Image.fromarray(equalized_img_array)

# Save or display the equalized image
equalized_img.save("./equalized_image.png")
equalized_img.show()

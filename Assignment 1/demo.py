from PIL import Image
import numpy as np
from global_hist_eq import get_equalization_transform_of_img


filename = "input_img.png"
img = Image.open(fp=filename)
bw_img = img.convert("L")
img_array = np.array(bw_img)

equalization_transform = get_equalization_transform_of_img(img_array)
print(equalization_transform)
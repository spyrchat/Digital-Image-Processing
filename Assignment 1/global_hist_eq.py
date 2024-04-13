import numpy as np
from PIL import Image

def get_equalization_transform_of_img(img_array: np.ndarray,):
    L = 256
    img_array = img_array.flatten()
    prob = np.zeros(L)
    for i in img_array:
        if i < 0 or i > L-1:
            raise ValueError("Image should be in the range of 0-255")
        prob[i] += 1

    prob = prob / len(img_array)
    # print (img_array.shape)
    u = np.zeros(256)
    u = np.cumsum(prob)
    y = np.round((u - u.min())/(1 - u.min()) * (L - 1))
    y = y.astype(np.uint8)
    return y

def perform_global_hist_equalization(img_array: np.ndarray):
    equalized_img = np.zeros_like(img_array)
    equalized_img = get_equalization_transform_of_img(img_array)
    equalized_img = equalized_img[img_array]
    return equalized_img

# img = perform_global_hist_equalization(np.array(Image.open("Assignment 1/input_img.png").convert("L")))
# img = Image.fromarray(img)
# img.show()

        


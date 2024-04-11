import numpy as np

def get_equalization_transform_of_img(img_array: np.ndarray,):
    L = 256
    img_array = img_array.flatten()
    prob = np.zeros(L)
    for i in img_array:
        if i < 0 or i > L-1:
            raise ValueError("Image should be in the range of 0-255")
        prob[i] += 1

    prob = prob / len(img_array)

    u = np.zeros(256)
    u = np.cumsum(prob)
    y = np.round((u - u[0])/(1 - u[0]) * (L - 1))
    return y
        


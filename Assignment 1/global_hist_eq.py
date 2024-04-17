import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

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
    y = np.round((u - u.min())/(1 - u.min()) * (L - 1))
    y = y.astype(np.uint8)
    return y

def perform_global_hist_equalization(img_array: np.ndarray):
    equalized_img = np.zeros_like(img_array)
    equalized_img = get_equalization_transform_of_img(img_array)
    equalized_img = equalized_img[img_array]
    return equalized_img

def plot_transformation_function(img_path):
    # Load the image and convert to grayscale
    img = Image.open(img_path).convert('L')
    img_array = np.array(img)

    # Get the transformation function using the defined function
    transformation = get_equalization_transform_of_img(img_array)

    # Generate a plot of the transformation function
    plt.figure(figsize=(8, 6))
    plt.plot(transformation, marker= None, linestyle='-')
    plt.title('Histogram Equalization Transformation Function')
    plt.xlabel('Original Intensity')
    plt.ylabel('Transformed Intensity')
    plt.grid(True)
    plt.show()

def get_histogram(img_array: np.ndarray):
    L = 256
    img_array = img_array.flatten()
    prob = np.zeros(L)
    for i in img_array:
        if i < 0 or i > L-1:
            raise ValueError("Image should be in the range of 0-255")
        prob[i] += 1
    return prob

def plot_histogram(img_array: np.ndarray):
    L = 256
    plt.figure(figsize=(12, 6))
    equalization_transform = perform_global_hist_equalization(img_array)
    y1 = get_histogram(img_array)
    y2 = get_histogram(equalization_transform)
    # Plot original histogram
    plt.subplot(1, 2, 1)
    plt.bar(np.arange(L), y1, color='blue')
    plt.title('Original Histogram')
    plt.xlabel('Intensity Value')
    plt.ylabel('Frequency')

    # Plot equalized histogram
    plt.subplot(1, 2, 2)
    plt.bar(np.arange(L), y2, color='red')
    plt.title('Equalized Histogram')
    plt.xlabel('Intensity Value')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()


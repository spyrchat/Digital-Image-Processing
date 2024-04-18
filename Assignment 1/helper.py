import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from global_hist_eq import get_equalization_transform_of_img, perform_global_hist_equalization
from adaptive_hist_eq import perform_adaptive_hist_equalization, perform_adaptive_hist_equalization_no_interpolation


#================================================================================================#
# The functions below are used to plot the transformation function and histogram of the image
# after global histogram equalization. They are helper functions for the demo script.
#================================================================================================#

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


def plot_histogram(img_array: np.ndarray, Title: str = 'Original Histogram', region_len_h: int = 48, region_len_w: int = 64):
    L = 256
    plt.figure(figsize=(12, 6))

    
    if Title == 'Global Equalized Histogram':
        equalization_transform = perform_global_hist_equalization(img_array)
        y = get_histogram(equalization_transform)
    elif Title == 'Adaptive Equalized Histogram':
        equalization_transform = perform_adaptive_hist_equalization(img_array, region_len_h, region_len_w)
        y = get_histogram(equalization_transform)
    elif Title == 'Original Histogram':
        y = get_histogram(img_array)
    elif Title == 'Adaptive Equalized Histogram_no_interpolation':
        equalization_transform = perform_adaptive_hist_equalization_no_interpolation(img_array, region_len_h, region_len_w)
        y = get_histogram(equalization_transform)

   
    # Plot original histogram
    
    plt.bar(np.arange(L), y, color='blue')
    plt.title(Title)
    plt.xlabel('Intensity Value')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

def plot_histogram_side_to_side(img_array: np.ndarray):
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
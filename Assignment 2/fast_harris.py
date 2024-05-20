import numpy as np
from PIL import Image
from skimage import io, color, feature, transform
# from scipy.ndimage import sobel, convolve
import matplotlib.pyplot as plt

def my_convolve(array: np.ndarray, kernel: np.ndarray):
    """Perform a 2D convolution without using any libraries."""
    # Get the dimensions of the image and kernel
    array_height, array_width = array.shape
    kernel_height, kernel_width = kernel.shape
    
    # Calculate the padding dimensions
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    
    # Pad the image with zeros around the border
    padded_image = np.pad(array, ((pad_height, pad_height), (pad_width, pad_width)), mode='reflect')
    
    # Initialize the output image
    output = np.zeros_like(array)
    
    # Perform the convolution
    for i in range(array_height):
        for j in range(array_width):
            # Extract the region of interest
            region = padded_image[i:i+kernel_height, j:j+kernel_width]
            
            # Perform element-wise multiplication and sum the result
            output[i, j] = np.sum(region * kernel)
    
    return output

def my_gaussian_kernel(size: int, sigma: float):
    """Generates a Gaussian kernel."""
    kernel = np.zeros((size, size))
    mean = size // 2
    sum_val = 0.0

    for i in range(size):
        for j in range(size):
            kernel[i, j] = (1 / (2 * np.pi * sigma ** 2)) * np.exp(
                -((i - mean) ** 2 + (j - mean) ** 2) / (2 * sigma ** 2)
            )
            sum_val += kernel[i, j]

    # Normalize the kernel
    kernel /= sum_val
    
    return kernel

def my_sobel(image: np.ndarray):
    # Sobel operators for x and y gradients
    G_i = np.array([[-1, 0, 1], 
                    [-2, 0, 2], 
                    [-1, 0, 1]])
    G_j = np.array([[-1, -2, -1], 
                    [ 0,  0,  0], 
                    [ 1,  2,  1]])
    
    I_i = my_convolve(image, G_i)
    I_j = my_convolve(image, G_j)
    
    return I_i, I_j

def my_corner_harris(img: np.ndarray, k: float, sigma: float) -> np.ndarray:

    # I_i = I1, I_j = I2
    I_i, I_j = my_sobel(img)

    # Computing Gaussian Kernel for size = round(4 * sigma)
    size = round(4 * sigma)
    kernel = my_gaussian_kernel(size, sigma)

    # Computing Elements of A matrix
    A_ii = I_i ** 2
    A_jj = I_j ** 2
    A_ij = I_i * I_j

    # Computing Elements of M matrix
    M_ii = my_convolve(A_ii, kernel)
    M_jj = my_convolve(A_jj, kernel)
    M_ij = my_convolve(A_ij, kernel)

    # Computing R matrix
    detM = (M_ii * M_jj) - (M_ij ** 2)
    traceM = M_ii + M_jj

    R = detM - k * (traceM ** 2)

    return R

def my_corner_peaks(harris_response: np.ndarray, rel_threshold: float):
    Height, Width = harris_response.shape

    threshold = harris_response.max() * rel_threshold
    points = []

    for i in range(Height):
        for j in range(Width):
            if harris_response[i, j] > threshold:
                points.append([i, j])

    return np.array(points)


if __name__ == "__main__":

    # Load the grayscale image
    filename = "images/im2.jpg"
    img = Image.open(fp=filename)
    img = img.resize((510, 660))
    grayscale_image = img.convert("L")
    grayscale_image_array = np.array(grayscale_image) / 255.0

    # Applying Harris Response
    harris_response = my_corner_harris(grayscale_image_array, k=0.04, sigma=2.0)
    corner_coords = my_corner_peaks(harris_response, rel_threshold=0.01)

    # Plot Image with Detected Corners
    plt.imshow(grayscale_image_array, cmap='gray')
    plt.title('Harris with Detected Corners')
    plt.scatter(corner_coords[:, 1], corner_coords[:, 0], color='red', marker='s', s=1)
    plt.show()

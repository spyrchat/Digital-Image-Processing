import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from PIL import Image

def my_corner_harris(img, k=0.04, sigma=1.0):
    gaussian_window = int(np.round(4 * sigma))
    H,W = np.shape(img)
    I1, I2 = compute_gradients(img)
    I11 = I1**2
    I12 = I1 * I2
    I22 = I2**2
    # Initialize the response matrix
    # Initialize the response matrix
    R = np.zeros((H, W))

    # Define the range for the Gaussian window
    half_window = gaussian_window // 2
    u = np.arange(-half_window, half_window + 1)

    # Create a grid for u1 and u2
    u1, u2 = np.meshgrid(u, u)

    # Calculate the Gaussian weights
    w = np.exp(- (u1**2 + u2**2) / (2 * sigma**2))

    # Pad the images to handle the border effects
    padded_I1 = np.pad(I1, half_window, mode='edge')
    padded_I2 = np.pad(I2, half_window, mode='edge')
    for p1 in range(H):
        for p2 in range(W):
           M = np.zeros((2, 2))
           for u1 in range(-gaussian_window // 2, gaussian_window // 2 + 1):
                for u2 in range(-gaussian_window // 2, gaussian_window // 2 + 1):
                    if 0 <= p1 + u1 < H and 0 <= p2 + u2 < W:
                        w = np.exp(- (u1**2 + u2**2) / (2 * sigma**2))
                        A11 = I11[p1 + u1, p2 + u2]
                        A12 = I12[p1 + u1, p2 + u2]
                        A21 = A12
                        A22 = I22[p1 + u1, p2 + u2]
                        A = np.array([[A11, A12], [A21, A22]])
                        M += w * A 
                det_M = M[0,0]*M[1,1] - M[0,1]*M[1,0]
                trace_M = M[0,0] + M[1,1]
                R[p1,p2] = det_M - k * trace_M**2
    return R

def compute_gradients(img):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1,-2, -1], [0, 0, 0], [1, 2, 1]])
    
    I1 = convolve(img, sobel_x)
    I2 = convolve(img, sobel_y)
    
    return I1, I2

def my_corner_peaks(harris_response, rel_threshold=0.1):
    threshold = harris_response.max() * rel_threshold
    H, W = harris_response.shape
    corner_locations = []

    for p1 in range(H):
        for p2 in range(W):
            if harris_response[p1, p2] > threshold:
                corner_locations.append((p1, p2))

    return np.array(corner_locations)

# Example usage
if __name__ == "__main__":
    img_path = 'Assignment 2/im2.jpg'
    img = Image.open(fp=img_path)
    img = img.resize((510, 660))
    # Keep only the Luminance component of the image
    img_gray = img.convert("L")

    # Obtain the underlying np array
    image = np.array(img_gray)

    # Normalize the image to the range [0, 1]
    image = image / 255.0

    R = my_corner_harris(image, k=0.04, sigma=2.0)
    corners = my_corner_peaks(R, rel_threshold=0.005)

    plt.imshow(image, cmap='gray')
    if corners.size > 0:  # Ensure corners array is not empty before plotting
        plt.scatter(corners[:, 1], corners[:, 0], c='r', s=5)
    plt.title('Harris Corners')
    plt.show()
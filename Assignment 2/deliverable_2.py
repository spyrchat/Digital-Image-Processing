import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import cv2

def compute_gradients(img):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    
    I1 = convolve(img, sobel_x)
    I2 = convolve(img, sobel_y)
    
    return I1, I2

def gaussian_weight(x1, x2, sigma):
    return np.exp(- (x1**2 + x2**2) / (2 * sigma**2))

def my_corner_harris(img, k=0.04, sigma=1.0):
    I1, I2 = compute_gradients(img)
    
    # Compute the second moment matrix components
    Ixx = I1**2
    Iyy = I2**2
    Ixy = I1 * I2
    
    # Dimensions of the image
    H, W = img.shape

    # Initialize the weighted sums
    Mxx = np.zeros((H, W))
    Myy = np.zeros((H, W))
    Mxy = np.zeros((H, W))

    window_size = int(4 * sigma + 1)
    half_window = window_size // 2

    for u1 in range(-half_window, half_window + 1):
        for u2 in range(-half_window, half_window + 1):
            w = gaussian_weight(u1, u2, sigma)
            shifted_Ixx = np.roll(Ixx, shift=(u1, u2), axis=(0, 1))
            shifted_Iyy = np.roll(Iyy, shift=(u1, u2), axis=(0, 1))
            shifted_Ixy = np.roll(Ixy, shift=(u1, u2), axis=(0, 1))
            Mxx += w * shifted_Ixx
            Myy += w * shifted_Iyy
            Mxy += w * shifted_Ixy
    
    # Compute the determinant and trace of the matrix M
    det_M = Mxx * Myy - Mxy**2
    trace_M = Mxx + Myy
    
    # Compute the Harris response
    R = det_M - k * (trace_M**2)
    
    return R

def my_corner_peaks(harris_response, rel_threshold=0.2):
    threshold = harris_response.max() * rel_threshold
    W = harris_response.shape[1]
    H = harris_response.shape[0]
    corner_locations = []

    for p1 in range(H):
        for p2 in range(W):
            if harris_response[p1, p2] > threshold:
                # Check if this is a local maximum
                if (harris_response[p1, p2] == np.max(harris_response[max(0, p1-1):min(H, p1+2), max(0, p2-1):min(W, p2+2)])):
                    corner_locations.append((p1, p2))

    return np.array(corner_locations)

# Example usage
if __name__ == "__main__":
    img_path = 'Assignment 2/im2.jpg'   
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Check if the image is loaded correctly
    if img is None:
        raise FileNotFoundError("The image file could not be loaded. Please check the path and the file.")
    print("Image loaded successfully.")

    # Optionally reduce the resolution to speed up processing
    img = cv2.resize(img, (img.shape[1] // 8, img.shape[0] // 8))

    R = my_corner_harris(img, k=0.04, sigma=1.0)
    corners = my_corner_peaks(R, rel_threshold=0.2)
    
    plt.imshow(img, cmap='gray')
    plt.scatter(corners[:, 1], corners[:, 0], c='r', s=5)
    plt.title('Harris Corners')
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import cv2

def my_corner_harris(img, k=0.04, sigma=1.0):
    gaussian_window = int(np.round(4 * sigma))
    
    W = img.shape[1]
    H = img.shape[0]
    R = np.zeros((H, W))
    I1, I2 = compute_gradients(img)

    for p1 in range(H):
        for p2 in range(W):
            M = np.zeros((2, 2))
            for u1 in range(-gaussian_window // 2, gaussian_window // 2 + 1):
                for u2 in range(-gaussian_window // 2, gaussian_window // 2 + 1):
                    if 0 <= p1 + u1 < H and 0 <= p2 + u2 < W:
                        w = np.exp(- (u1**2 + u2**2) / (2 * sigma**2))
                        A11 = I1[p1 + u1, p2 + u2]**2
                        A12 = I1[p1 + u1, p2 + u2] * I2[p1 + u1, p2 + u2]
                        A21 = A12
                        A22 = I2[p1 + u1, p2 + u2]**2
                        A = np.array([[A11, A12], [A21, A22]])
                        M += w * A
            R[p1, p2] = np.linalg.det(M) - k * (np.trace(M)**2)
    return R

def compute_gradients(img):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    
    I1 = convolve(img, sobel_x)
    I2 = convolve(img, sobel_y)
    
    return I1, I2

def my_corner_peaks(harris_response, rel_threshold=0.1):
    threshold = harris_response.max() * rel_threshold
    W = harris_response.shape[1]
    H = harris_response.shape[0]
    corner_locations = []

    for p1 in range(H):
        for p2 in range(W):
            if harris_response[p1, p2] > threshold:
                # Check if this is a local maximum
                local_max = True
                for i in range(max(0, p1-1), min(H, p1+2)):
                    for j in range(max(0, p2-1), min(W, p2+2)):
                        if harris_response[i, j] > harris_response[p1, p2]:
                            local_max = False
                            break
                    if not local_max:
                        break
                if local_max:
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
img_grayscale = cv2.resize(img, (img.shape[1] // 8, img.shape[0] // 8))

if img.ndim == 3:
    img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])

    R = my_corner_harris(img, k=0.04, sigma=1.0)
    corners = my_corner_peaks(R, rel_threshold=0.1)
    
    plt.imshow(img, cmap='gray')
    plt.scatter(corners[:, 1], corners[:, 0], c='r', s=5)
    plt.title('Harris Corners')
    plt.show()
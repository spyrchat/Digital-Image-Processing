import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, convolve
from PIL import Image
import cv2


#The function my_corner_harris can get Ridiculously SLOW! especially for larger images, so I have implemented the algorithm the way the excircise commanded 
#And I also implemented a method that is the default and it is named 'Fast' that uses a Gaussian filter directly to the Gradient Product Images Ixx Ixy and Iyy
#Essentially it performs the same Weighted summation over a neighborhood BUT it runs orders of Magnitude Faster. I left the other method that can be tried if you pass 
#######'ByTheBook'##### as an argument in my_corner_harris.
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, convolve
from PIL import Image

def my_corner_harris(img, k=0.04, sigma=1.0, method='Fast'):
    if method == 'Fast':
        H, W = img.shape
        I1, I2 = compute_gradients(img)
        
        # Pre-compute I11, I12, I22
        I11 = I1**2
        I12 = I1 * I2
        I22 = I2**2
        
        # Apply Gaussian filter to the squared gradients
        S11 = gaussian_filter(I11, sigma=sigma) #The default value for the Gaussian Window is 4σ
        S12 = gaussian_filter(I12, sigma=sigma)
        S22 = gaussian_filter(I22, sigma=sigma)
        
        # Compute the response matrix R
        det_M = S11 * S22 - S12**2
        trace_M = S11 + S22
        R = det_M - k * trace_M**2
        return R
    
    if method == 'ByTheBook':
        gaussian_window = int(np.round(4 * sigma))
        H, W = np.shape(img)
        I1, I2 = compute_gradients(img)
        # Pre-compute I11, I12, I22 to save computation time
        I11 = I1**2
        I12 = I1 * I2
        I22 = I2**2
        # Initialize the response matrix
        R = np.zeros((H, W))

        # Define the range for the Gaussian window
        half_window = gaussian_window // 2
        u = np.arange(-half_window, half_window + 1)

        # Create a grid for u1 and u2
        u1, u2 = np.meshgrid(u, u)

        # Calculate the Gaussian weights
        w = np.exp(- (u1**2 + u2**2) / (2 * sigma**2))
        # Find the Harris response as it was presented in the exercise
        for p1 in range(H):
            for p2 in range(W):
                M = np.zeros((2, 2))
                for u1 in range(-gaussian_window // 2, gaussian_window // 2 + 1):
                    for u2 in range(-gaussian_window // 2, gaussian_window // 2 + 1):
                        if 0 <= p1 + u1 < H and 0 <= p2 + u2 < W:
                            A11 = I11[p1 + u1, p2 + u2]
                            A12 = I12[p1 + u1, p2 + u2]
                            A21 = A12
                            A22 = I22[p1 + u1, p2 + u2]
                            A = np.array([[A11, A12], [A21, A22]])
                            M += w * A 
                # Calculate the determinant of M
                det_M = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
                # Calculate the trace of M
                trace_M = M[0, 0] + M[1, 1]
                # Get the Harris response that we will later threshold to decide if the pixel is a corner
                R[p1, p2] = det_M - k * trace_M**2
        return R

def compute_gradients(img):
    # We can approximate the partial derivatives with the use of a high pass mask
    # Here the Sobel approximation is used
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    I1 = convolve(img, sobel_x)
    I2 = convolve(img, sobel_y)
    
    return I1, I2

def my_corner_peaks(harris_response, rel_threshold=0.1):
    # Compute the absolute threshold value
    threshold = harris_response.max() * rel_threshold
    # Create a boolean mask where the Harris response exceeds the threshold
    mask = harris_response > threshold
    # Get the coordinates of the points where the mask is True
    corner_locations = np.argwhere(mask)
    return corner_locations

# Example usage
if __name__ == "__main__":
    img_path = 'Assignment 2/im1.jpg'
    img = Image.open(fp=img_path)
    img_high_res_rgb = np.array(img)
    img_high_res = img.convert("L")
    img_high_res = np.array(img_high_res)

    # Resize the image for lower resolution processing
    scale_factor = 0.1  # Scale factor for lower resolution
    img_low_res = cv2.resize(img_high_res, (int(img_high_res.shape[1] * scale_factor), int(img_high_res.shape[0] * scale_factor)))

    # Normalize the low resolution image to the range [0, 1]
    img_low_res = img_low_res / 255.0

    # Compute Harris response on the low resolution image
    R = my_corner_harris(img_low_res, k=0.05, sigma=3.0, method='Fast')

    # Detect corners on the low resolution image
    corners = my_corner_peaks(R, rel_threshold=0.005)

    # Rescale the corner coordinates to the original high-resolution image
    corners_high_res = (corners / scale_factor).astype(int)

    plt.imshow(img_high_res_rgb, cmap='gray')
    if corners_high_res.size > 0:  # Ensure corners array is not empty before plotting
        plt.scatter(corners_high_res[:, 1], corners_high_res[:, 0], c='r', s=5)
    plt.title('Harris Corners on High-resolution Image')
    plt.show()
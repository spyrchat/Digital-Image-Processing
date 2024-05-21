import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, convolve
from PIL import Image


#The function my_corner_harris can get Ridiculously SLOW! especially for larger images, so I have implemented the algorithm the way the excircise commanded 
#And I also implemented a method that is the default and it is named 'Fast' that aookies a Gaussian filter directly to the Gradient Product Images Ixx Ixy and Iyy
#Essentially it performs the same Weighted summation over a neighborhood BUT it runs orders of Magnitude Faster. I left the other method that can be tried if you pass 
#######'ByTheBook'##### as an argument in my_corner_harris.
def my_corner_harris(img, k=0.04, sigma=1.0, method='Fast'):
    if(method == 'Fast'):
        H, W = img.shape
        I1, I2 = compute_gradients(img)
        
        # Pre-compute I11, I12, I22
        I11 = I1**2
        I12 = I1 * I2
        I22 = I2**2
        
        # Apply Gaussian filter to the squared gradients
        gaussian_window = int(np.round(4 * sigma))
        half_window = gaussian_window // 2

        S11 = gaussian_filter(I11, sigma=sigma)
        S12 = gaussian_filter(I12, sigma=sigma)
        S22 = gaussian_filter(I22, sigma=sigma)
        
        # Compute the response matrix R
        det_M = S11 * S22 - S12**2
        trace_M = S11 + S22
        R = det_M - k * trace_M**2
        return R
    
    if(method == 'ByTheBook' ):
        gaussian_window = int(np.round(4 * sigma))
        H,W = np.shape(img)
        I1, I2 = compute_gradients(img)
        #Pre Compute I11, I12, I22 to save computation Time
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
        # Find The Harris Response as it was presented in the Exercise
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
                        #Calculate The Determinant of M
                        det_M = M[0,0]*M[1,1] - M[0,1]*M[1,0]
                        #Calculate The Trace of M
                        trace_M = M[0,0] + M[1,1]
                        #Get the Harris Response That we will later Threshold to decide if the pixel is a corner
                        R[p1,p2] = det_M - k * trace_M**2
        return R

def compute_gradients(img):
    # We can Approximate the partial Derivatives with the use a High Pass Mask
    # Here the Sobel Approximation is used
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1,-2, -1], [0, 0, 0], [1, 2, 1]])
    
    I1 = convolve(img, sobel_x)
    I2 = convolve(img, sobel_y)
    
    return I1, I2

# This Function Essentially decides given the harris response if the 
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
    img_path = 'Assignment 2/im3.jpg'
    img = Image.open(fp=img_path)
    img = img.resize((510, 660))
    img = img.convert("L")
    # Obtain the underlying np array
    image = np.array(img)

    # Normalize the image to the range [0, 1]
    image = image / 255.0

    R = my_corner_harris(image, k=0.04, sigma=2.0,method='Fast')
    corners = my_corner_peaks(R, rel_threshold=0.005)

    plt.imshow(image, cmap='gray')
    if corners.size > 0:  # Ensure corners array is not empty before plotting
        plt.scatter(corners[:, 1], corners[:, 0], c='r', s=5)
    plt.title('Harris Corners')
    plt.show()
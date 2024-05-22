import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
#When an image is rotated, its corners may extend beyond the original bounds, necessitating a larger canvas to fit the entire rotated image.
def calculate_new_dimensions(width, height, angle):
    # Calculate the new width of the rotated image
    # The new width is determined by projecting the original width and height onto the rotated axes
    new_width = int(abs(width * np.cos(angle)) + abs(height * np.sin(angle)))
    # Calculate the new height of the rotated image
    # The new height is determined by projecting the original width and height onto the rotated axes
    new_height = int(abs(width * np.sin(angle)) + abs(height * np.cos(angle)))
    return new_width, new_height

#Performs the Interpolation with the 4 neighboring pixel values(If there are 4 if there arent then it 
#skips them and just takes only the neighbors that are inside the image borders
def bilinear_interpolate(img, x, y, C):
    H = img.shape[0]
    W = img.shape[1]
    if C > 1:
        temp = np.zeros(3)
    else:
        temp = 0
    #Count Exists because I want to find the average of the neighboring pixel values and for pixels that are in the borders there arent 4 neighbors. 
    #Therefore this way the algorithm can skip neibors that fall out of the image dimensions or do not exist.
    count = 0

    if y + 1 < H and y + 1 >= 0 and x < W and x >= 0:
        temp += img[y + 1, x]
        count = count + 1
    if x - 1 < W and x - 1 >= 0 and y < H and y >= 0:
        temp += img[y, x - 1]
        count = count + 1
    if y - 1 < H and y - 1 >= 0 and x < W and x >= 0:
        temp += img[y - 1, x]
        count = count + 1
    if x + 1 < W and x + 1 >= 0 and y < H and y >= 0:
        temp += img[y, x + 1]
        count = count + 1
    interpolated_value = temp // count

    return interpolated_value
#### The Angle should Be Given In RADIANS #####
def my_img_rotation(img, angle):
    H, W = img.shape[:2]
    C = 1 if img.ndim == 2 else img.shape[2]
    new_W, new_H = calculate_new_dimensions(W, H, angle)
    rotated_img = np.zeros((new_H, new_W), dtype=img.dtype) if C == 1 else np.zeros((new_H, new_W, C), dtype=img.dtype)
    
    cx, cy = W // 2, H // 2
    new_cx, new_cy = new_W // 2, new_H // 2

    # Create meshgrid for coordinates
    x, y = np.meshgrid(np.arange(new_W), np.arange(new_H))

    # Flatten the arrays for vectorized operations
    x_flat = x.flatten()
    y_flat = y.flatten()

    # Apply counterclockwise rotation
    x_og = ((x_flat - new_cx) * np.cos(angle) - (y_flat - new_cy) * np.sin(angle) + cx).astype(int)
    y_og = ((x_flat - new_cx) * np.sin(angle) + (y_flat - new_cy) * np.cos(angle) + cy).astype(int)

    # Check for valid coordinates
    valid_coords = (0 <= x_og) & (x_og < W) & (0 <= y_og) & (y_og < H)

    # Perform bilinear interpolation for valid coordinates
    if C == 1:
        rotated_img[y_flat[valid_coords], x_flat[valid_coords]] = img[y_og[valid_coords], x_og[valid_coords]]
    else:
        rotated_img[y_flat[valid_coords], x_flat[valid_coords], :] = img[y_og[valid_coords], x_og[valid_coords], :]

    return rotated_img

def main(image_path, angle):
    # Load the image
    img = Image.open(image_path)
    # img = img.resize((510, 660)) 
    #img = img.convert("L") # The algorithm works for both RGB and Grayscale Images
    img_array = np.array(img)
    
    # Rotate the image
    rotated_img_array = my_img_rotation(img_array, angle) 
    rotated_img_array = rotated_img_array / 255

    # Display the original and rotated images
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    ax[0].set_title("Original Image")
    ax[0].imshow(img_array, cmap='gray')

    ax[1].set_title(f"Rotated Image ({angle*(180/np.pi)}°)")
    ax[1].imshow(rotated_img_array, cmap='gray') 

    plt.show()

if __name__ == "__main__":
    image_path = 'Assignment 2\im2.jpg'  # Path to the uploaded image
    main(image_path, 54*(np.pi/180))
    main(image_path, 213*(np.pi/180))
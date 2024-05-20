import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def calculate_new_dimensions(width, height, angle):
    angle = np.deg2rad(angle)
    new_width = int(abs(width * np.cos(angle)) + abs(height * np.sin(angle)))
    new_height = int(abs(width * np.sin(angle)) + abs(height * np.cos(angle)))
    return new_width, new_height

def bilinear_interpolate(img, x, y):
    H, W = img.shape[:2]
    x1, y1 = int(np.floor(x)), int(np.floor(y))
    x2, y2 = min(x1 + 1, W - 1), min(y1 + 1, H - 1)
    
    if x1 < 0 or x2 >= W or y1 < 0 or y2 >= H:
        return 0
    
    Q11 = img[y1, x1]
    Q21 = img[y1, x2]
    Q12 = img[y2, x1]
    Q22 = img[y2, x2]
    
    R1 = (x2 - x) * Q11 + (x - x1) * Q21
    R2 = (x2 - x) * Q12 + (x - x1) * Q22
    P = (y2 - y) * R1 + (y - y1) * R2
    
    return P

def rot_img(img, angle):
    H, W = img.shape[:2]
    C = 1 if img.ndim == 2 else img.shape[2]

    new_W, new_H = calculate_new_dimensions(W, H, angle)
    rotated_img = np.zeros((new_H, new_W), dtype=img.dtype) if C == 1 else np.zeros((new_H, new_W, C), dtype=img.dtype)
    
    cx, cy = W // 2, H // 2
    new_cx, new_cy = new_W // 2, new_H // 2
    
    angle_rad = np.deg2rad(angle)
    
    for y in range(new_H):
        for x in range(new_W):
            x_og = (x - new_cx) * np.cos(angle_rad) - (y - new_cy) * np.sin(angle_rad) + cx
            y_og = (x - new_cx) * np.sin(angle_rad) + (y - new_cy) * np.cos(angle_rad) + cy
            
            if 0 <= x_og < W and 0 <= y_og < H:
                rotated_img[y, x] = bilinear_interpolate(img, x_og, y_og)

    return rotated_img

def main(image_path, angle):
    # Load the image
    img = Image.open(image_path)  # Ensure grayscale
    img = img.resize((510, 660))  
    img_array = np.array(img)
    
    # Rotate the image
    rotated_img_array = rot_img(img_array, angle)  # Counterclockwise rotation
    rotated_img_array = rotated_img_array / 255

    # Display the original and rotated images
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    ax[0].set_title("Original Image")
    ax[0].imshow(img_array, cmap='gray')

    ax[1].set_title(f"Rotated Image ({angle}°)")
    ax[1].imshow(rotated_img_array, cmap='gray')

    plt.show()

if __name__ == "__main__":
    image_path = 'Assignment 2/im2.jpg'  # Path to the uploaded image
    main(image_path, 54)
    main(image_path, 213)
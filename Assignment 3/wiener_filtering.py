import numpy as np

def my_wiener_filter(y: np.ndarray, h: np.ndarray, K: float) -> np.ndarray:
    # height_y, width_y = y.shape
    # height_h, width_h = h.shape

    # # Determine the shape for convolution
    # conv_shape = [height_y + height_h - 1, width_y + width_h - 1]

    # # Pad the h and y arrays to the determined shape
    # padded_h = np.pad(h, ((0, conv_shape[0] - height_h), (0, conv_shape[1] - width_h)), mode='constant', constant_values=0)
    # padded_y = np.pad(y, ((0, conv_shape[0] - height_y), (0, conv_shape[1] - width_y)), mode='constant', constant_values=0)


    # # Compute the Fourier transforms
    # H = np.fft.fft2(padded_h, s=conv_shape)
    # Y = np.fft.fft2(padded_y, s=conv_shape)

    # # Compute the Wiener filter
    # H_conj = np.conj(H)
    # H_abs_squared = np.abs(H) ** 2
    # Wiener_filter = H_conj / (H_abs_squared + 1/K)

    # # Apply the Wiener filter to the observed image in the frequency domain
    # G = Wiener_filter * Y

    # # Compute the inverse Fourier transform to get the filtered image
    # x_hat = np.fft.ifft2(G)

    # # Take the real part and crop to the original image size
    # x_hat = np.real(x_hat[:height_y, :width_y])
    # return x_hat

    H = np.fft.fft2(h, s=y.shape)
    H = np.conj(H) / (np.abs(H) ** 2 + 1/K)
    Y = np.fft.fft2(y)
    X_hat = H * Y
    x_hat = np.fft.ifft2(X_hat).real
    return x_hat

    

import numpy as np


def my_wiener_filter(y: np.ndarray, h: np.ndarray, K: float) -> np.ndarray:
    height_y, width_y = y.shape
    height_h, width_h = h.shape

    # Determine the shape for convolution
    conv_shape = [height_y + height_h - 1, width_y + width_h - 1]

    padded_h = np.pad(h, ((0, conv_shape[0] - height_h), (0, conv_shape[1] - width_h)), mode='constant', constant_values=0)
    padded_y = np.pad(y, ((0, conv_shape[0] - height_y), (0, conv_shape[1] - width_y)), mode='constant', constant_values=0)

    H = np.fft.fft2(padded_h)
    Y = np.fft.fft2(padded_y)
    H_conj = np.conj(H)
    H_abs_squared = np.abs(H) ** 2
    Wiener_filter = H_conj / (H_abs_squared + 1/K)

    G = Wiener_filter * Y
    x_hat = np.fft.ifft2(G)
    x_hat = np.real(x_hat[:height_y, :width_y])
    return x_hat



    

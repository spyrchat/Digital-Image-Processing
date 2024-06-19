import numpy as np


def my_wiener_filter(y: np.ndarray,h: np.ndarray,K: float):
    height_y = np.shape[0]
    width_y = np.shape[1]
    height_h = np.shape[0]
    width_h = np.shape[1]

    conv_shape = []
    conv_shape[0] = height_y + height_h - 1
    conv_shape[1] = width_h + width_y - 1

    padded_h = np.pad(h, ((0, conv_shape[0] - height_h), (0, conv_shape[1] - width_h)), mode='constant', constant_values=0)
    padded_y = np.pad(y, ((0, conv_shape[0] - height_y), (0, conv_shape[1] - width_y)), mode='constant', constant_values=0)

    H = np.fft.fft2(padded_h)
    Y = np.fft.fft2(padded_y)
    H_conj = np.conj(H)
    H_abs_squared = np.abs(H) ** 2
    Wiener_filter = H_conj / (H_abs_squared + 1/K)

    G = Wiener_filter * Y
    
    

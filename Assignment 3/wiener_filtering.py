import numpy as np

def my_wiener_filter(y: np.ndarray, h: np.ndarray, K: float) -> np.ndarray:
    #Perform the Fast Fourier Transfrom of h after zero padding to match the dimensions of image y
    H = np.fft.fft2(h, s=y.shape)
    #G is the Wiener Filter. K is a hyperparameter that has a crucial role in the filter's performance.
    G = np.conj(H) / (np.abs(H) ** 2 + 1/K)
    #Perform the Fast Fourier Transfrom of y
    Y = np.fft.fft2(y)
    #Multiply the Fourier Transfrom of Y with the Wiener Filter. Convolution is multiplication in the Frequency Domain.
    X_hat = G * Y
    # x_hat is the output of the filter and it as an estimation of the original image. Perfrom the Inverse Fourier Transfrom so we can plot the output image.
    x_hat = np.fft.ifft2(X_hat).real
    return x_hat

    
def inverse_H_filter(y: np.ndarray, h: np.ndarray) -> np.ndarray:
    # Fourier domain noise free image inverse filter
    H = np.fft.fft2(h, s=y.shape)
    epsilon = 1e-10  # Small value to avoid division by zero
    Y = np.fft.fft2(y)
    X_inv = (1 / (H + epsilon)) * Y
    x_inv = np.fft.ifft2(X_inv).real
    return x_inv


import numpy as np

def my_wiener_filter(y: np.ndarray, h: np.ndarray, K: float) -> np.ndarray:
    H = np.fft.fft2(h, s=y.shape)
    G = np.conj(H) / (np.abs(H) ** 2 + 1/K)
    Y = np.fft.fft2(y)
    X_hat = G * Y
    x_hat = np.fft.ifft2(X_hat).real
    return x_hat

    

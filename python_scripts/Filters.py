import cv2
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

# customized imshow function


def imshow(img, cap=None):
    if np.amax(img) > 255:
        img = img / (np.amax(img)) * 255
    img.astype(np.uint8)
    fig = plt.figure(figsize=(4, 4))
    if cap is not None:
        plt.title(cap)
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.show()


def apply(img, ker):
    return np.abs(np.fft.ifft2(np.fft.fftshift(np.fft.fft2(img)) * ker).real)

################################################################

# Ideal LowPass Filter kernel


def ilpf(m, n, r):
    a = np.tile(np.arange(-n / 2, n / 2), (m, 1))
    b = np.tile(np.arange(-m / 2, m / 2), (n, 1)).T
    return (a * a + b * b < r * r).astype(np.uint8)

# Print ILPF Result


def apply_ilpf(img, r):
    row, col = img.shape
    imshow(img(*ilpf(row, col, r)), 'Ideal LowPass Filter\nwith r = ' + str(r))


img = cv2.imread("C:/Users/NGPD/Desktop/2.jpg", cv2.IMREAD_GRAYSCALE)
imshow(img, 'Loaded Image')
apply_ilpf(img, 5)
apply_ilpf(img, 15)
apply_ilpf(img, 30)
apply_ilpf(img, 80)

################################################################

# Butterworth LPF kernel


def blpf(m, n, N, r):
    a = np.tile(np.arange(-n / 2, n / 2), (m, 1))
    b = np.tile(np.arange(-m / 2, m / 2), (n, 1)).T
    return (1 / (1 + ((a * a + b * b) / (r * r))**N))

# Print BLPF Result


def apply_blpf(img, N, r):
    row, col = img.shape
    imshow(img(*blpf(row, col, N, r)),
           'Butterworth LowPass Filter\nwith r = ' + str(r) + ' and n = ' + str(N))


img = cv2.imread("C:/Users/NGPD/Desktop/2.jpg", cv2.IMREAD_GRAYSCALE)
imshow(img, 'Loaded Image')
n = 2
apply_blpf(img, n, 5)
apply_blpf(img, n, 15)
apply_blpf(img, n, 30)
apply_blpf(img, n, 80)

################################################################

# Gaussian LPF kernel


def glpf(m, n, r):
    a = np.tile(np.arange(-n / 2, n / 2), (m, 1))
    b = np.tile(np.arange(-m / 2, m / 2), (n, 1)).T
    return np.exp(-(a * a + b * b) / (2 * r * r))

# Print GLPF Result


def apply_glpf(img, r):
    row, col = img.shape
    imshow(img(*glpf(row, col, r)), 'Gaussian LowPass Filter\nwith r = ' + str(r))


img = cv2.imread("C:/Users/NGPD/Desktop/2.jpg", cv2.IMREAD_GRAYSCALE)
imshow(img, 'Loaded Image')
apply_glpf(img, 5)
apply_glpf(img, 15)
apply_glpf(img, 30)
apply_glpf(img, 80)

################################################################

img = cv2.imread("C:/Users/NGPD/Desktop/2.jpg", cv2.IMREAD_GRAYSCALE)

m, n = img.shape
a = np.tile(np.arange(-n / 2, n / 2), (m, 1))
b = np.tile(np.arange(-m / 2, m / 2), (n, 1)).T
lap = img(*-(a * a + b * b))
if np.amax(lap) > 255:
    lap = lap / (np.amax(lap)) * 255

imshow(img, 'Loaded Image')
imshow(lap, 'Laplacian Filter')
imshow(img - lap, 'g(x, y)')

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


def ilpf(m, n, r):
    a = np.tile(np.arange(-n / 2, n / 2), (m, 1))
    b = np.tile(np.arange(-m / 2, m / 2), (n, 1)).T
    return (a * a + b * b < r * r).astype(np.uint8)


def apply(img, ker):
    return np.abs(np.fft.ifft2(np.fft.fftshift(np.fft.fft2(img)) * ker).real)


# generate chess board with size NxN
N = 64
a = np.tile(np.array(1), (int(N / 4), int(N / 4)))
b = np.tile(np.array(0), (int(N / 4), int(N / 4)))
img1 = np.concatenate([a, b], axis=0)
img2 = np.concatenate([b, a], axis=0)
img = np.concatenate([img1, img2], axis=1)
img = np.tile(img, (4, 4))

r, c = img.shape
ker = ilpf(r, c, 30)
res = img(*ker)

imshow(img, 'Created Image')
imshow(ker, 'ILPF Kernel')
imshow(res, 'ILPF Result')

import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt


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


def shift(array):
    n = array.shape[0]
    t = array[0:int(n / 2), 0:int(n / 2)].copy()
    array[0:int(n / 2), 0:int(n / 2)] = array[int(n / 2):n, int(n / 2):n]
    array[int(n / 2):n, int(n / 2):n] = t
    t = array[0:int(n / 2), int(n / 2):n].copy()
    array[0:int(n / 2), int(n / 2):n] = array[int(n / 2):n, 0:int(n / 2)]
    array[int(n / 2):n, 0:int(n / 2)] = t
    return array


def padding(img):
    s = 2**np.ceil(np.log2(np.amax(img.shape))).astype(np.int32)
    height = s - img.shape[0]
    width = s - img.shape[1]
    left = width // 2
    right = width - left
    top = height // 2
    down = height - top
    shape_ = [[top, down], [left, right]]
    return np.pad(img, shape_, 'constant')


def dft1d(array):
    N = array.shape[0]
    # (a[x, y], b[x, y]) = (x, y)
    a = np.tile(np.arange(0, N), (N, 1))
    b = a.copy().T
    W = np.exp(-2j * np.pi / N * a * b)
    return np.around(np.dot(W, array), 2)


def idft1d(array):
    N = array.shape[0]
    # (a[x, y], b[x, y]) = (x, y)
    a = np.tile(np.arange(0, N), (N, 1))
    b = a.copy().T
    W = np.exp(2j * np.pi / N * a * b) / N
    return np.around(np.dot(W, array), 2)


def fft1d(array):
    m = array.shape[0]
    if m == 1:
        return array
    elif m % 2 == 0:
        even = fft1d(array[::2])
        odd = fft1d(array[1::2])
        Wm = np.exp(-2j * np.pi / m * np.arange(int(m / 2)))
        half1 = even + odd * Wm
        half2 = even - odd * Wm
        return np.concatenate([half1, half2])
    else:
        raise ValueError("Wrong dimension")


def fft(array):
    n = array.shape[0]
    if np.log2(n) % 1 != 0:
        return dft1d(array)
    else:
        return np.around(fft1d(array), 2)


def fft2d(matrix):
    temp = np.array([fft(x) for x in matrix]).T
    return np.around(np.array([fft(x) for x in temp]).T, 2)

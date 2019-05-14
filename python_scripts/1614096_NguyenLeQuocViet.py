from skimage.exposure import rescale_intensity
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math
def convolve(image, kernel):
	(iH, iW) = image.shape[:2]
	(kH, kW) = kernel.shape[:2]
	kernel = kernel[::-1]
	pad = (kW - 1) // 2
	image = cv.copyMakeBorder(image, pad, pad, pad, pad, cv.BORDER_REPLICATE)
	output = np.zeros((iH,iW), dtype="float32")

	for y in np.arange(pad, iH + pad):
		for x in np.arange(pad, iW + pad):
			roi = image[y - pad:y+pad+1, x - pad:x + pad + 1]
			k =(roi * kernel).sum()
			output[y - pad, x - pad] = k
		
	output = rescale_intensity(output, in_range=(0,255))
	output = (output * 255).astype("uint8")
	return output

I = np.array([  [5,0,0,1,2],
                [2,1,5,1,2],
                [7,1,5,1,2],
                [7,4,5,4,3],
                [7,1,6,1,3] ])
    
sobelX = np.array((
	[-1,0,1],
	[-2,0,2],
	[-1,0,1]), dtype="int")

sobelY = np.array((
	[-1,-1,-1],
	[-2, 0, 2],
	[-1, 0, 1]), dtype="int")

H,W = I.shape
D_x = convolve(I,sobelX)
D_y = convolve(I,sobelY)
result_a = []
for i in range(H):
    temp = []
    for j in range(W):
        temp += [(D_x[i,j],D_y[i,j])]
    result_a.append(temp)
print("1A\n",result_a)

def create_histogram(img):
    assert len(img.shape) == 2
    H,W = img.shape
    sum = H * W
    histogram = np.zeros(shape=(8,), dtype = float)
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            histogram[img[row,col]] += 1 / sum
    return histogram


def visualize_histogram(histogram, name):
    index = np.arange(len(histogram))
    plt.bar(index,histogram)
    plt.xlabel('Intensity', fontsize = 5)
    plt.ylabel('Frequency', fontsize = 5)
    plt.title(name)
    plt.show()

def histogram_equation(histogram, img):
    c = np.cumsum(histogram)
    print(c)
    m_table = np.array([]).astype(np.uint8)
    m_table = c * 7
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            img[row,col] = m_table[img[row,col]]
    return img


visualize_histogram(create_histogram(I),"1B")

print("1C\n",histogram_equation(create_histogram(I),I))

def DFT1D(array):
    N = array.shape[0]
    # (a[x, y], b[x, y]) = (x, y)
    a = np.tile(np.arange(0, N), (N, 1))
    b = a.copy().T
    W = np.exp(-2j*np.pi/N*a*b)
    return np.around(np.dot(W, array), 2)

def DCT1D(array):
    N = array.shape[0]
    factor = math.pi / N
    C = np.zeros((N, N), dtype = np.float32)
    for x in range(N):
        C[0][x] = math.sqrt(1/N) * math.cos((x + 0.5) * 0 * factor)
    for u in range(N)[1:]:
        for x in range(N):
            C[u][x] = math.sqrt(2/N) * math.cos((x + 0.5) * u * factor)
    return C, np.matmul(C, array)

print("2A\n",DFT1D(np.array([1,3])))
C, F = DCT1D(np.array([1,0,1,0]))
print("2B\nC=\n", C)
print("F=",F)

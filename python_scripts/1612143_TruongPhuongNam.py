import cv2 as cv
import numpy as np
from numpy import fft
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy import signal
from scipy import fftpack

def create_histogram(img):
	size = img.shape[0]*img.shape[1]
	hist = np.zeros(8, dtype=np.float)
	for row in range(img.shape[0]):
		for col in range(img.shape[1]):
			hist[img[row][col]] += 1
	return hist/size

def create_sum_histogram(hist):
	sum_hist = np.zeros(8,dtype=np.float)
	sum_hist[0] = hist[0]
	for x in range(1,8):
		sum_hist[x] = sum_hist[x-1] + hist[x]
	return sum_hist

def dft(signal):
	N = len(signal)
	x = np.array(range(N))
	x = np.tile(x,(N,1))
	u = x.copy().T
	M = np.exp((-1j*2*np.pi*x*u)/N)
	return np.matmul(M,signal)

def show_dft_vector(N):
	x = np.array(range(N))
	x = np.tile(x,(N,1))
	u = x.copy().T
	M = np.exp((-1j*2*np.pi*x*u)/N)
	print(M)

def dct(signal):
	N = len(signal)
	x = np.array(range(N))
	x = np.tile(x,(N,1))
	u = x.copy().T
	M = np.cos((2*x+1)*u*np.pi/(2*N))
	a1 = [np.sqrt(1/N)]
	a2 = np.repeat(np.array([np.sqrt(2/N)]),N-1)
	a = np.concatenate((a1,a2),axis = 0)
	return np.matmul(M,signal)*a

def show_dct(N):
	x = np.array(range(N))
	x = np.tile(x,(N,1))
	u = x.copy().T
	M = np.cos((2*x+1)*u*np.pi/(2*N))
	a1 = [np.sqrt(1/N)]
	a2 = np.repeat(np.array([np.sqrt(2/N)]),N-1)
	a = np.concatenate((a1,a2),axis = 0)
	a = np.repeat(a,4)
	a = np.reshape(a,(4,4))
	return M*a


def bai1():
	I = np.array([[5,0,0,1,2],
				[2,1,5,1,2],
				[7,1,5,1,2],
				[7,4,5,4,3],
				[7,1,6,1,3]])
	Kx = np.array([[-1,0,1],
				[-2,0,2],
				[-1,0,1]])
	Ky = Kx.T
	Gx = ndimage.convolve(I,Kx,mode='reflect')
	Gy = ndimage.convolve(I,Ky,mode='reflect')
	print('Gradient vector at (0,0): (' + str(Gx[0,0]) + ',' + str(Gy[0,0]) +')')
	print('Gradient vector at (1,1): (' + str(Gx[1,1]) + ',' + str(Gy[1,1]) +')')
	print('Gradient vector at (0,3): (' + str(Gx[0,3]) + ',' + str(Gy[0,3]) +')')

	h = create_histogram(I)
	plt.plot(h,color = 'b')
	plt.xlim([0,7])
	plt.show()
	sumhist = create_sum_histogram(h)

	new_I = I.copy()
	for x in range(I.shape[0]):
		for y in range(I.shape[1]):
			new_I[x][y] = int(np.round(sumhist[I[x][y]]*7))
	new_h = create_histogram(new_I)
	new_h_sum = create_sum_histogram(new_h)
	plt.plot(sumhist,color = 'r')
	plt.plot(new_h_sum,color = 'b')
	plt.xlim([0,7])
	plt.show()

	print(new_I)


def bai2():
	# a)
	print('Cau 2a)')
	show_dft_vector(2)
	print(dft([1,3]))
	# b)
	print('Cau b)')
	fx = np.array([1,3])
	fu = dft(fx)
	print(fu)
	print('Chung minh truc giao chuan:')
	base_vector = show_dct(4)
	for v in base_vector:
		print(np.sum(v**2))
	for x in range(base_vector.shape[0]):
		for y in range(x,base_vector.shape[0]):
			if x == y:
				continue
			print(np.dot(base_vector[x],base_vector[y]))

	fx = np.array([1,0,1])
	fxx = np.array([1,0,1,0])
	print(dct(fxx))


def bai3():
	A = np.array([[1/np.sqrt(2),-1/np.sqrt(2),3],
				[1/np.sqrt(2),1/np.sqrt(2),-3*2**0.5+3],
				[0,0,1]])
	P1 = np.array([2,3,1])
	pixel = np.round(np.matmul(np.linalg.inv(A),P1))
	print(pixel)

bai1()
bai2()
bai3()
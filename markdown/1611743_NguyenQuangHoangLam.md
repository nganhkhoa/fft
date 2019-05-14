```{.python .input  n=84}
import numpy as np
from scipy import signal
from scipy import fftpack
import matplotlib.pyplot as plt 


print("Question 1: ")
### Question 1
print("a/")
##a 

I1 = np.array([
    [5,0,0,1,2],
    [2,1,5,1,2],
    [7,1,5,1,2],
    [7,4,5,4,3],
    [7,1,6,1,3]
])


H_sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
H_sobel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])

grad_x = signal.convolve2d(I1,H_sobel_x,boundary='symm',mode ='same')
grad_y = signal.convolve2d(I1,H_sobel_y,boundary='symm',mode = 'same')


print(f"Gradient of (0,0) according to x and y: {grad_x[0,0]} {grad_y[0,0]}")
print(f"Gradient of (1,1) according to x and y: {grad_x[1,1]} {grad_y[1,1]}")
print(f"Gradient of (0,3) according to x and y: {grad_x[0,3]} {grad_y[0,3]}")

##b
print("b/")
n = I1.size
hist1 = np.bincount(I1.flatten())/n
print("Normalized Histogram of I1: ")
print(hist1)

##c
print("c/")
hist2 = np.array([1/8 for i in range(8)])
cdf1 = np.cumsum(hist1)
cdf2 = np.cumsum(hist2)

## equalize 
print("I2: ")
I2 = np.zeros((N,N))
N = I1.shape[0]
for i in range(N):
  for j in range(N):
    x = I1[i,j]
    c_x = cdf1[x]
    for k in range(len(cdf2)):
      if cdf2[k] == c_x:
        I2[i,j] = k
        
print(I2.astype(np.int))

print("Histogram of I2: ")
plt.hist(I2.flatten(),8,[0,8])
plt.show()

    
### Question 2 
print("Question 2: ")
#a
print("a/")
f = np.array([1,2])

A = np.arange(2).reshape((1,2))

M = A.reshape((2,1)).dot(A)

M_dft = np.round(np.exp(1)**(-2j*(np.pi/2)*M))

print("Based vectors after DFT: ")
i=0
for e in M_dft:
  print(f"e{i}: {e}")
  i+=1
  
#b 
print("b/")
M = 4
u = np.arange(M).reshape((M,1))
x = 2*u+1
W_dct = np.dot(u,x.T)
W_dct = np.cos(np.pi*W_dct/(2*M))
for _ in range(M):
  if _ == 0:
    W_dct[_,:]*= np.sqrt(1/M)
  else:
    W_dct[_,:]*= np.sqrt(2/M)

print("Based vectors after DCT for 4 dims: ")
i=0
for e in W_dct:
  print(f"e{i}: {e}")
  i+=1
  
IW = np.round(W_dct.dot(W_dct.T)).astype(np.int32)
print(f"W_dct.dot(W_dct.T): \n {IW}")

if(np.array_equal(IW,np.eye(4))):
  print("W is orthor")


fx = np.array([1,0,1])

fx_dct = fftpack.dct(fx)

print(f"dct of fx: {fx_dct}")

### Question 3:

print("Question 3: ")
print("a/")
phi = np.pi/4
H2 = np.array([[np.cos(phi),-np.sin(phi),0],[np.sin(phi),np.cos(phi),0],[0,0,1]])
H1 = np.array([[1,0,-3],[0,1,-3],[0,0,1]])
H3 = np.array([[1,0,3],[0,1,3],[0,0,1]])

print(f"Translate -3: \n{H1}")
print(f"Rotate : \n {H2}")
print(f"Translate 3: \n{H3}")
H = H3.dot(H2).dot(H1)

print(f"Matrix to rotate the image around (3,3): \n {H}")

print("b/")

Io = np.array((2,3,1)).reshape((3,1))
I = np.linalg.pinv(H).dot(Io)

x,y=I[0:2,:].flatten()
print(f"x, y after zero order: {np.round(x).astype(np.int32)},{np.round(y).astype(np.int32)}")


```

```{.json .output n=84}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Question 1: \na/\nGradient of (0,0) according to x and y: 16 8\nGradient of (1,1) according to x and y: 1 -9\nGradient of (0,3) according to x and y: -3 -5\nb/\nNormalized Histogram of I1: \n[0.08 0.28 0.16 0.08 0.08 0.16 0.04 0.12]\nc/\nI2: \n[[0 0 0 0 0]\n [0 0 0 0 0]\n [7 0 0 0 0]\n [7 0 0 0 0]\n [7 0 0 0 0]]\nHistogram of I2: \n"
 },
 {
  "data": {
   "image/png": "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\nAABJRU5ErkJggg==\n",
   "text/plain": "<Figure size 432x288 with 1 Axes>"
  },
  "metadata": {
   "tags": []
  },
  "output_type": "display_data"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Question 2: \na/\nBased vectors after DFT: \ne0: [1.+0.j 1.+0.j]\ne1: [ 1.+0.j -1.-0.j]\nb/\nBased vectors after DCT for 4 dims: \ne0: [0.5 0.5 0.5 0.5]\ne1: [ 0.65328148  0.27059805 -0.27059805 -0.65328148]\ne2: [ 0.5 -0.5 -0.5  0.5]\ne3: [ 0.27059805 -0.65328148  0.65328148 -0.27059805]\nW_dct.dot(W_dct.T): \n [[1 0 0 0]\n [0 1 0 0]\n [0 0 1 0]\n [0 0 0 1]]\nW is orthor\ndct of fx: [4. 0. 2.]\nQuestion 3: \na/\nTranslate -3: \n[[ 1  0 -3]\n [ 0  1 -3]\n [ 0  0  1]]\nRotate : \n [[ 0.70710678 -0.70710678  0.        ]\n [ 0.70710678  0.70710678  0.        ]\n [ 0.          0.          1.        ]]\nTranslate 3: \n[[1 0 3]\n [0 1 3]\n [0 0 1]]\nMatrix to rotate the image around (3,3): \n [[ 0.70710678 -0.70710678  3.        ]\n [ 0.70710678  0.70710678 -1.24264069]\n [ 0.          0.          1.        ]]\nb/\nx, y after zero order: 2,4\n"
 }
]
```

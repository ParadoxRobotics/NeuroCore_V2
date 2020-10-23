# Local contrast  normalization implementation based on LeCun LCN.
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt

# create gaussian kernel (size = int and sigma = float)
def gaussian_kernel(size, sigma):
    side = int((size-1)//2)
    x, y = np.mgrid[-side:side+1, -side:side+1]
    g = np.exp(-(x**2 / (sigma**2 * float(side)) + y**2 / (sigma**2 * float(side))))
    return (g / np.sum(g)).astype('float32')

# simple local normalization
def Local_Contrast_Normalization(img, kernel):
    # get image size
    imgSize = img.shape
    cvImgSum = img.astype('float32') - cv2.filter2D(img, None, kernel)
    cvImgSqrt = cv2.filter2D(np.square(img), None, kernel)
    cvImgSigma = np.sqrt(cvImgSqrt)
    c = np.mean(cvImgSigma)
    return np.maximum(cvImgSum/c, cvImgSum/cvImgSigma)

img = cv2.imread('car1.jpg')
plt.imshow(img)
plt.show()

plt.imshow(gaussian_kernel(size=12, sigma=6))
plt.show()

y = Local_Contrast_Normalization(img, gaussian_kernel(size=5, sigma=6))
y = cv2.normalize(y, dst=y, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
plt.imshow(y)
plt.show()

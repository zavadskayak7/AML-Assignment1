# import packages: numpy, math (you might need pi for gaussian functions)
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import convolve2d as conv2



"""
Gaussian function taking as argument the standard deviation sigma
The filter should be defined for all integer values x in the range [-3sigma,3sigma]
The function should return the Gaussian values Gx computed at the indexes x
"""
def gauss(sigma):
    n_x = len(range(int(-3*sigma), int(3*sigma)+1)) # number of x points
    x = np.linspace(-3*sigma, 3*sigma, n_x)
    var = (x**2) / (2*sigma**2)
    Gx = (1 / (np.sqrt(2*math.pi) * sigma)) * np.exp(-var) # Gaussian formula for 1 variable
    return Gx, x.astype(int)



"""
Implement a 2D Gaussian filter, leveraging the previous gauss.
Implement the filter from scratch or leverage the convolve2D method (scipy.signal)
Leverage the separability of Gaussian filtering
Input: image, sigma (standard deviation)
Output: smoothed image
"""
def gaussianfilter(img, sigma):
    # 2D kernel made from applying twice convolution of 1D kernel
    Gx = gauss(sigma)[0] # 1D-x filter
    Gx = Gx.reshape(1, Gx.size)
    Gy = Gx.T # 1D-y filter
    # to speed up computations, decrease number of x points in gauss function
    smooth_img = conv2(conv2(img, Gx, 'same'), Gy, 'same')
    return smooth_img



"""
Gaussian derivative function taking as argument the standard deviation sigma
The filter should be defined for all integer values x in the range [-3sigma,3sigma]
The function should return the Gaussian derivative values Dx computed at the indexes x
"""
def gaussdx(sigma):
    n_x = len(range(int(-3*sigma), int(3*sigma)+1)) # number of x points
    x = np.linspace(-3*sigma, 3*sigma, n_x)
    var = (x**2) / (2*sigma**2)
    var1 = (1 / (np.sqrt(2*math.pi) * sigma**3))
    Dx = -var1 * x * np.exp(-var) # 1st derivative
    return Dx, x.astype(int)



def gaussderiv(img, sigma):
    Gx = gauss(sigma)[0] # Gaussian 1D filter
    Dx = gaussdx(sigma)[0] # Gaussian x-derivative 1D filter
    Gx = Gx.reshape(1, Gx.size)
    Dx = Dx.reshape(1, Gx.size)
    Dy = Dx.T # Gaussian y-derivative 1D filter

    # image smoothed with std sigma and derived in x and y directions
    imgDx = conv2(conv2(img, Gx, 'same'), Dx, 'same')
    imgDy = conv2(conv2(img, Gx, 'same'), Dy, 'same')
    return imgDx, imgDy


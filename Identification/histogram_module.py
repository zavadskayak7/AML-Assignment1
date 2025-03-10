import numpy as np
from numpy import histogram as hist



#Add the Filtering folder, to import the gauss_module.py file, where gaussderiv is defined (needed for dxdy_hist)
import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
filteringpath = os.path.join(parentdir, 'Filtering')
sys.path.insert(0,filteringpath)
import gauss_module



#  compute histogram of image intensities, histogram should be normalized so that sum of all values equals 1
#  assume that image intensity varies between 0 and 255
#
#  img_gray - input image in grayscale format
#  num_bins - number of bins in the histogram
def normalized_hist(img_gray, num_bins):
    assert len(img_gray.shape) == 2, 'image dimension mismatch'
    assert img_gray.dtype == 'float', 'incorrect image type'
    bins=np.arange(0,255+1,step=255/num_bins)
    counts=np.zeros(num_bins,dtype='int')

    for pixel in img_gray.reshape(128*128):
        for number,bin_ in enumerate(bins):
            if pixel>=bin_ and pixel < bins[number+1]:
            #print(pixel,number+1)
                counts[number]+=1
    
    hists= np.round(np.divide(counts,sum(counts)),2)
    


    return hists, bins



#  Compute the *joint* histogram for each color channel in the image
#  The histogram should be normalized so that sum of all values equals 1
#  Assume that values in each channel vary between 0 and 255
#
#  img_color - input color image
#  num_bins - number of bins used to discretize each channel, total number of bins in the histogram should be num_bins^3
#
#  E.g. hists[0,9,5] contains the number of image_color pixels such that:
#       - their R values fall in bin 0
#       - their G values fall in bin 9
#       - their B values fall in bin 5
def rgb_hist(img_color_double, num_bins):
    assert len(img_color_double.shape) == 3, 'image dimension mismatch'
    assert img_color_double.dtype == 'float', 'incorrect image type'


    bins=np.arange(0,255+1,step=255/num_bins)
    
    img_color_double_reshaped=img_color_double.reshape(128*128,3)  


    #Define a 3D histogram  with "num_bins^3" number of entries
    hists = np.zeros((num_bins, num_bins, num_bins))
    
    # Loop for each pixel i in the image 
    for i in range(img_color_double.shape[0]*img_color_double.shape[1]):
        # Increment the histogram bin which corresponds to the R,G,B value of the pixel i
        counts=np.zeros(img_color_double.shape[2],dtype='int')
        for j in range(img_color_double.shape[2]):
            for number,bin_ in enumerate(bins):
                if img_color_double_reshaped[i][j]>=bin_ and img_color_double_reshaped[i][j]<bins[number+1]:
                    counts[j]=number
        hists[counts[0],counts[1],counts[2]]+=1       


    #Normalize the histogram such that its integral (sum) is equal 1
    hists= np.divide(hists,np.sum(hists))

    #Return the histogram as a 1D vector
    hists = hists.reshape(hists.size)
    return hists



#  Compute the *joint* histogram for the R and G color channels in the image
#  The histogram should be normalized so that sum of all values equals 1
#  Assume that values in each channel vary between 0 and 255
#
#  img_color - input color image
#  num_bins - number of bins used to discretize each channel, total number of bins in the histogram should be num_bins^2
#
#  E.g. hists[0,9] contains the number of image_color pixels such that:
#       - their R values fall in bin 0
#       - their G values fall in bin 9
def rg_hist(img_color_double, num_bins):
    assert len(img_color_double.shape) == 3, 'image dimension mismatch'
    assert img_color_double.dtype == 'float', 'incorrect image type'


    bins=np.arange(0,255+1,step=255/num_bins)
    
    img_color_double=img_color_double.reshape(128*128,3)



    #Define a 2D histogram  with "num_bins^2" number of entries
    hists = np.zeros((num_bins, num_bins))
    
    
    # Loop for each pixel i in the image 
    for i in range(img_color_double.shape[0]):
        # Increment the histogram bin which corresponds to the R,G value of the pixel i
        counts=np.zeros(img_color_double.shape[1]-1,dtype='int')
        for j in range(img_color_double.shape[1]-1):
            for number,bin_ in enumerate(bins):
                if img_color_double[i][j]>=bin_ and img_color_double[i][j]<bins[number+1]:
                    counts[j]=number
        hists[counts[0],counts[1]]+=1


    #Normalize the histogram such that its integral (sum) is equal 1
    hists= np.divide(hists,np.sum(hists))


    #Return the histogram as a 1D vector
    hists = hists.reshape(hists.size)

    return hists




#  Compute the *joint* histogram of Gaussian partial derivatives of the image in x and y direction
#  Set sigma to 3.0 and cap the range of derivative values is in the range [-6, 6]
#  The histogram should be normalized so that sum of all values equals 1
#
#  img_gray - input gray value image
#  num_bins - number of bins used to discretize each dimension, total number of bins in the histogram should be num_bins^2
#
#  Note: you may use the function gaussderiv from the Filtering exercise (gauss_module.py)
def dxdy_hist(img_gray, num_bins):
    assert len(img_gray.shape) == 2, 'image dimension mismatch'
    assert img_gray.dtype == 'float', 'incorrect image type'


    minimo,maximo=-6,6


    #Define a 2D histogram  with "num_bins^2" number of entries
    hists = np.zeros((num_bins, num_bins))


    [imgDx,imgDy]=gauss_module.gaussderiv(img_gray,sigma=3)
       
    bins=np.linspace(minimo,maximo+1,num_bins)
    
    imgDx=np.clip(imgDx,minimo,maximo)
                                                        # cap values to be in range [-6,6]
    imgDy=np.clip(imgDy,minimo,maximo)
             
    
    imgDx=imgDx.reshape(imgDx.size)
    imgDy=imgDy.reshape(imgDy.size)
    
    
    assert len(imgDx)==len(imgDy)
    
    for i in range(len(imgDx)): 

        counts=np.zeros(len(img_gray.shape),dtype='int')
        for number,bin_ in enumerate(bins):
            if imgDx[i]>=bin_ and imgDx[i] < bins[number+1]:
                counts[0]=number
            if imgDy[i]>=bin_ and imgDy[i] < bins[number+1]:   
                counts[1]=number
                
        hists[counts[0],counts[1]]+=1
        
    hists= np.divide(hists,np.sum(hists))


    #Return the histogram as a 1D vector
    hists = hists.reshape(hists.size)
    return hists



def is_grayvalue_hist(hist_name):
  if hist_name == 'grayvalue' or hist_name == 'dxdy':
    return True
  elif hist_name == 'rgb' or hist_name == 'rg':
    return False
  else:
    assert False, 'unknown histogram type'


def get_hist_by_name(img, num_bins_gray, hist_name):
  if hist_name == 'grayvalue':
    return normalized_hist(img, num_bins_gray)
  elif hist_name == 'rgb':
    return rgb_hist(img, num_bins_gray)
  elif hist_name == 'rg':
    return rg_hist(img, num_bins_gray)
  elif hist_name == 'dxdy':
    return dxdy_hist(img, num_bins_gray)
  else:
    assert False, 'unknown distance: %s'%hist_name


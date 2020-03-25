import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import heapq

import histogram_module
import dist_module

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray



# model_images - list of file names of model images
# query_images - list of file names of query images
#
# dist_type - string which specifies distance type:  'chi2', 'l2', 'intersect'
# hist_type - string which specifies histogram type:  'grayvalue', 'dxdy', 'rgb', 'rg'
#
# note: use functions 'get_dist_by_name', 'get_hist_by_name' and 'is_grayvalue_hist' to obtain 
#       handles to distance and histogram functions, and to find out whether histogram function 
#       expects grayvalue or color image

def find_best_match(model_images, query_images, dist_type, hist_type, num_bins):

    hist_isgray = histogram_module.is_grayvalue_hist(hist_type)
    
    model_hists = compute_histograms(model_images, hist_type, hist_isgray, num_bins)
    query_hists = compute_histograms(query_images, hist_type, hist_isgray, num_bins)
    
    D = np.zeros((len(model_images), len(query_images)))
    
    
    best_match=[]

    
    for i in range(len(query_images)):
        for j in range(len(model_images)):
            D[j,i]=dist_module.get_dist_by_name(model_hists[j],query_hists[i],dist_type)
        minpos = list(D[:,i]).index(min(list(D[:,i]))) 
        best_match.append(int(model_images[minpos].split('/')[1].split('_')[0].split('obj')[1]))



    return best_match, D



def compute_histograms(image_list, hist_type, hist_isgray, num_bins):
    
    image_hist = []

    # Compute hisgoram for each image and add it at the bottom of image_hist

    image_hist = []
    
    for i in range(len(image_list)):
        img= np.array(Image.open(image_list[i]))

        if histogram_module.is_grayvalue_hist(hist_type)==False:        
            img= img.astype('double')
            
        elif histogram_module.is_grayvalue_hist(hist_type)==True:
            img = rgb2gray(img.astype('double'))
            
        hist = histogram_module.get_hist_by_name(img, num_bins, hist_type)
        if len(hist) == 2: # !!!important  the normalized histogram(greyvalue name) function returns the bins too, so they are cut at this part
            hist = hist[0]
        image_hist.append(hist)


    return image_hist



# For each image file from 'query_images' find and visualize the 5 nearest images from 'model_image'.
#
# Note: use the previously implemented function 'find_best_match'
# Note: use subplot command to show all the images in the same Python figure, one row per query image

def show_neighbors(model_images, query_images, dist_type, hist_type, num_bins):
    
    
    plt.figure()

    num_nearest = 5  # show the top-5 neighbors
    
    best_match,D=find_best_match(model_images,query_images,dist_type,hist_type,num_bins)
    
    for i in range(len(query_images)):
        img_query= np.array(Image.open(query_images[i]))
        plt.subplot(1,num_nearest+1,1)
        plt.imshow(img_query)
        plt.title('Q'+str(i))



 
        for j in range(1,num_nearest+1):
            minpos = list(D[:,i]).index(heapq.nsmallest(j,(list(D[:,i])))[-1])
            img_match= np.array(Image.open(model_images[minpos]))
            plt.subplot(1,num_nearest+1,1+j)
            plt.imshow(img_match)
            plt.title('M0.'+str(round(heapq.nsmallest(j,(list(D[:,i])))[-1],2)))



            
            
            

        plt.show()


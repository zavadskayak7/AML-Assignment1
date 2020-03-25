import numpy as np
import math



# Compute the intersection distance between histograms x and y
# Return 1 - hist_intersection, so smaller values correspond to more similar histograms
# Check that the distance range in [0,1]

def dist_intersect(x,y):
    
    assert len(x)==len(y)
    sum_q_v=0
    sum_q=0
    sum_v=0
    for i in range(len(y)):
        #print(x[i],y[i])
        sum_q_v+=min(x[i],y[i])
        sum_q+=x[i]
        sum_v+=y[i]
    
    sim=(sum_q_v/sum_q+sum_q_v/sum_v)/2
    assert 0<=sim<=1
    return 1-sim



# Compute the L2 distance between x and y histograms
# Check that the distance range in [0,sqrt(2)]

def dist_l2(x,y):
    
    assert len(x)==len(y)
    sums=0
    for i in range(len(x)):
        sums+=np.square(x[i]-y[i])
    assert 0<=sums<=math.sqrt(2),str(sums)
    return sums



# Compute chi2 distance between x and y
# Check that the distance range in [0,Inf]
# Add a minimum score to each cell of the histograms (e.g. 1) to avoid division by 0

def dist_chi2(x,y):
    
    assert len(x)==len(y)
    sums=0
    for i in range(len(x)):
        x[i]+=1

        y[i]+=1
        sums+=np.square(x[i]-y[i])/(x[i]+y[i])
    assert 0<=sums<=math.inf,str(sums)
    return sums



def get_dist_by_name(x, y, dist_name):
  if dist_name == 'chi2':
    return dist_chi2(x,y)
  elif dist_name == 'intersect':
    return dist_intersect(x,y)
  elif dist_name == 'l2':
    return dist_l2(x,y)
  else:
    assert False, 'unknown distance: %s'%dist_name
  





import numpy as np
import matplotlib.pyplot as plt

import histogram_module
import dist_module
import match_module



# compute and plot the recall/precision curve
#
# D - square matrix, D(i, j) = distance between model image i, and query image j
#
# note: assume that query and model images are in the same order, i.e. correct answer for i-th query image is the i-th model image

def plot_rpc(D, plot_color):
    
    recall = []
    precision = []
    num_queries = D.shape[1]
    
    num_images = D.shape[0]
    assert(num_images == num_queries), 'Distance matrix should be a squatrix'
    
    labels = np.diag([1]*num_images)
      
    d = D.reshape(D.size)
    l = labels.reshape(labels.size)
     
    sortidx = d.argsort()
    d = d[sortidx]
    l = l[sortidx]
    number_of_threshold_values=200 # you can put even higher number, for a less 'straight' but more robust curve
    threshold_values=np.linspace(start=np.min(D),stop=np.max(D),num=number_of_threshold_values) #histograms-distance threshold

    for threshold in threshold_values:
    
        tp =0
        fp=0
        fn=0


        for idt in range(len(d)):
            if l[idt]==1 and d[idt]<=threshold:
                tp+=l[idt]
            elif l[idt]==1 and d[idt]>threshold:
                fn+=l[idt]
            elif l[idt]==0 and d[idt]<=threshold:
                fp+=1
    

        
        #Compute precision and recall values and append them to "recall" and "precision" vectors
        precision.append(tp/(tp+fp))
        recall.append(tp/(tp+fn))
    
    plt.plot([1-precision[i] for i in range(len(precision))], recall, plot_color+'-')
    plt.xlabel('1-precison')
    plt.ylabel('recall')



def compare_dist_rpc(model_images, query_images, dist_types, hist_type, num_bins, plot_colors):
    
    assert len(plot_colors) == len(dist_types), 'number of distance types should match the requested plot colors'

    for idx in range( len(dist_types) ):

        [best_match, D] = match_module.find_best_match(model_images, query_images, dist_types[idx], hist_type, num_bins)

        plot_rpc(D, plot_colors[idx])
    

    plt.axis([0, 1, 0, 1]);
    plt.xlabel('1 - precision');
    plt.ylabel('recall');
    
    # legend(dist_types, 'Location', 'Best')
    
    plt.legend( dist_types, loc='best')






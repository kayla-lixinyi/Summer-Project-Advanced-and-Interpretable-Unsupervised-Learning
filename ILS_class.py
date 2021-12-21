import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import argrelmax
from scipy.ndimage.filters import gaussian_filter1d
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import distance_metrics
from sklearn.manifold import TSNE

class ILS():
    """ Iterative Label Spreading

    Parameters
    ----------

    n_clusters : int, default=None
        The number of clusters expected to be identified
        from given dataset.

    min_cluster_size : int, default=None
        The minimum number of data points to be considered as a cluster.

    metric : String, default='euclidean'
        The valid metric for pairwise_distance.
        Must be a metric listed in pairwise.PAIRWISE_DISTANCE_FUNCTIONS or
        an option allowed by scipy.spatial.distance.pdist

    Attributes
    ----------

    dataset : ndarray of shape(n_samples, n_features)
        Transform input dataset into numpy ndarray

    rmin : ndarray of shape(n_samples, )
        The R_min distance of each iteration

    Examples
    ---------
    ##to be added after implementation of find peaks algorithm

    """
    def __init__(self, n_clusters = None, min_cluster_size = None, metric = 'euclidean', significance = 3.4):

        self.n_clusters = n_clusters # need to calculate defaults based on data set input
        self.min_cluster_size = min_cluster_size
        self.metric = metric
        self.significance = significance

    def fit(self, X):
        '''
        Main ILS function. Run iteravtive label spreading first to identify peaks to find points of highest density.
        Finally label points given the initial points found from points of highest density.
        INPUT:
            X = data set to be clustered
                numpy 2D array or pandas data frame
        OUTPUT:
            self = returns the ILS object where clustering results are stored.
        '''

        if self.min_cluster_size is None and self.n_clusters is None:
            self.min_cluster_size = int (0.05 * X.shape[0])
        elif self.min_cluster_size is None:
            self.min_cluster_size = int (X.shape[0]/(self.n_clusters * 2)) #currently assumes maximum of 20 clusters

        self.data_set = np.concatenate((np.array(X), np.zeros((X.shape[0],1))), axis = 1)
        self.rmin = []

        self.data_set[0, X.shape[1]] = 1 #initialise first label
        unlabelled = [i + 1 for i in range(X.shape[0] - 1)] # step 1
        
        label_spreading = self.label_spreading([0], unlabelled)
        
        new_centers, new_unlabelled = self.find_initial_points() # step 2
        
        label_spreading = self.label_spreading(new_centers, new_unlabelled, first_run = False) #step 3

        return self
    
    def find_minima(self):
        
        index = np.arange(len(self.rmin))
        
        pks = self.find_maxima_forward(self.rmin, [], 0)
        reved = self.find_maxima_forward(self.rmin[::-1], [], 0)
        non_rev = [len(self.rmin) - reved[i] - 1 for i in range(len(reved))]
        pks = pks + non_rev[::-1]
        pks = np.sort(pks)
        
        pks, n_SD = self.split_disconnected(self.rmin, pks)
        
        maxima, max_SD = self.get_final(pks, self.min_cluster_size//4 * 3, n_SD)
        
        if not self.n_clusters is None:
            try:
                inds = np.argsort(max_SD)[-self.n_clusters+1:]
                maxima = np.sort([maxima[i] for i in inds]).tolist()
            except:
                raise Exception("Only {} clusters were found. If you do not know how many clusters there are do not specify a number")
        
        filtered = gaussian_filter1d(self.rmin, max(2, self.min_cluster_size//32))
        
        betweenMax = np.split(filtered, maxima)
        betweenIndex = np.split(index, maxima)  
        
        minima = [(np.argpartition(betweenMax[i], 4)[:4] + 1 + betweenIndex[i][0]).tolist() for i in range(len(betweenMax))]

        return minima
    
    def find_maxima_forward(self, rmin, pks, check_peak):
    
        if check_peak + self.min_cluster_size > len(rmin) - 1:
            return pks

        sublst = rmin[check_peak:check_peak + self.min_cluster_size]

        maxind = np.argmax(sublst)

        if maxind == 0:
            pks.append(check_peak)
            return self.find_maxima_forward(rmin, pks, check_peak+1)
        else:
            return self.find_maxima_forward(rmin, pks, check_peak+1)
    
    def split_disconnected(self, rmin, maxs):

        peaks = []
        num_standev = []
        
        for i in range(maxs.shape[0]):
            if maxs[i] < self.min_cluster_size or maxs[i] > len(rmin) - self.min_cluster_size: 
                continue
                
            mean = np.mean(rmin[maxs[i] - self.min_cluster_size:maxs[i]])
            standev = np.var(rmin[maxs[i] - self.min_cluster_size:maxs[i]]) ** 0.5
            
            mean1 = np.mean(rmin[maxs[i]:maxs[i]+self.min_cluster_size])
            standev1 = np.var(rmin[maxs[i]:maxs[i]+self.min_cluster_size]) ** 0.5
            
            signifdiff = rmin[maxs[i]] > mean + self.significance * standev
            signifdiff1 = rmin[maxs[i]] > mean1 + self.significance * standev1
            
            if signifdiff or signifdiff1: 
                num_standev.append(max((rmin[maxs[i]] - mean)/standev, (rmin[maxs[i]] - mean1)/standev1))
                peaks.append(maxs[i])
                
        return peaks, num_standev
        
    def get_final(self, peaks, width_const, SD):
    
        width = width_const

        fin_peaks = []
        fin_SD = []
        ind = 0
        pks_temp = [peaks[0]]
        SD_temp = [SD[0]]

        for i in range(len(peaks)):
            if i == len(peaks) - 1:
                continue
            if peaks[i+1] - peaks[i] < width:
                pks_temp.append(peaks[i+1])
                SD_temp.append(SD[i+1])
                width = width - peaks[i+1] + peaks[i]
            else:
                fin_peaks.append(pks_temp)
                fin_SD.append(SD_temp)
                pks_temp = [peaks[i+1]]
                SD_temp = [SD[i+1]]
                width = width_const

        fin_peaks.append(pks_temp)
        fin_SD.append(SD_temp)

        fin_ind = [np.argmax(i) for i in fin_SD]
        fin_peaks = [fin_peaks[i][fin_ind[i]] for i in range(len(fin_ind))]

        fin_max = [np.max(i) for i in fin_SD]

        return fin_peaks, fin_max    

    def find_initial_points(self):
        '''
        Finds the points of highest density within the clusters found in the initial run and labels them in seperate classes.
        OUTPUTS:
            labelled_points: indices of labelled points (points of highest density)
                list of integers
            unlabelled_points: indices of unlabelled points (points of highest density)
                list of integers
        '''

        # get points of maximum density
        labelled_points = self.find_minima()

        counter = 1

        # label them in the data_set
        for i in labelled_points:
            for j in i:
                self.data_set[self.indOrdering[j], -1] = counter
            counter += 1
        
        labels = []
        
        for i in labelled_points:
            labels = labels + i

        labelled_points = self.indOrdering[labels]

        unlabelled_points = [i for i in range(self.data_set.shape[0]) if not i in labelled_points]

        # check that we haven't missed any points
        if len(unlabelled_points) + len(labelled_points) != self.data_set.shape[0]:
            raise Exception("The number of labelled (0) and unlabelled (0) points does not sum to the total (100) in find_initial_points")

        return labelled_points, unlabelled_points
    
    def create_colr(self, lsts):
        colour = ['red', 'blue', 'gray', 'black', 'orange', 'purple', 'green', 'yellow', 'brown', 'red', 'blue', 'gray', 'black', 'orange', 'purple', 'green', 'yellow', 'brown']
        return [colour[i] for i in lsts]
    
    def tSNE(self):
        
        points = self.data_set[:, :-1].copy()
        
        points_embedding = TSNE(metric = self.metric).fit(points)
        
        return points_embedding.embedding_      
        
    
    def predict(self, points):
        '''
        predict takes in an array of points and returns an 1D array of there corresponding labels
        INPUTS:
            points = 2D array of floats
        OUTPUTS:
            labels = 1D array of labels in the same order as th points given.
        '''
        
        D = pairwise_distances(self.data_set[:, :-1], 
                             points,
                             metric = self.metric)
        
        label_points = [np.unravel_index(D[:, i].argmin(), D[:, i].shape) for i in range(points.shape[0])]
        label_points = [label_points[i][0] for i in range(len(label_points))]
        
        return self.data_set[label_points, -1]
    
    def coloured_rmin(self):
        colour = ['red', 'blue', 'gray', 'black', 'orange', 'purple', 'green', 'yellow', 'brown', 'red', 'blue', 'gray', 'black', 'orange', 'purple', 'green', 'yellow', 'brown']
        
        for i in range(len(self.rmin)-2):
            plt.plot([i, i+1], self.rmin[i:i+2], color = colour[self.data_set[self.indOrdering.astype(int)[i], -1].astype(int)], linewidth = 0.8)
        
        plt.show()
    
    def plot_labels(self):
        
        if self.data_set.shape[1] - 1 > 2:
            data_set = np.concatenate((self.tSNE(), self.data_set[:, -1].reshape((-1,1))), axis = 1)
        else:
            data_set = self.data_set
        
        plt.scatter(data_set[:, 0], data_set[:, 1], color = self.create_colr(data_set[:, -1].astype(int)),s = 2)
        plt.show()

    def label_spreading(self, labelled_points, unlabelled_points, first_run = True):
        """
        Written by Amanda Parker
        Applies iterative label spreading to the given points
        
        Parameters
        ----------
        labelled_points : ndarray of shape(labelled_num, n_features+1)
            Initial points that are already labelled, the last column indicates the label of the point. 
            0 indicates unlabelled points, positive integers indicate labelled points.
            
        unlabelled_points = ndarray of shape(unlabelled_num, n_features+1)
            Initial points that are not labelled, the last column indicates the label of the point. 
            0 indicates unlabelled points, positive integers indicate labelled points.
            
        Returns
        ---------
        closest : ndarray of shape(n_samples-1, 2)
            Each row indicates the point(in Column 2) where the point's(in Column 1) label is spread from
            Both columns represent points as indices from 
        """
        indices = np.arange(self.data_set.shape[0]) 
        oldIndex = np.arange(self.data_set.shape[0]) 
        
        indOrdering = np.array(labelled_points).reshape(-1,)
        oldIndOrdering = np.array(unlabelled_points).reshape(-1,)
       
        labelled = self.data_set[labelled_points]
        unlabelled = self.data_set[unlabelled_points]
      
        labelColumn = self.data_set.shape[1]-1
        # lists for ordered output data
        outD = []
        outID = []
        closeID = []
    
        count = 0
        # Continue to label data points until all data points are labelled
        while len(unlabelled) > 0 :
            # Calculate labelled to unlabelled distances matrix (D) 
            D = pairwise_distances(
                labelled[:, :-1],
                unlabelled[:, :-1], metric=self.metric)
            # Find the minimum distance between a labelled and unlabelled point
            # first the argument in the D matrix
            (posL, posUnL) = np.unravel_index(D.argmin(), D.shape)
            #append R_min distance
            if first_run:
                self.rmin.append(D.min())
                
            indOrdering = np.concatenate((indOrdering, oldIndOrdering[posUnL].reshape(1,)), axis = 0)
            oldIndOrdering = np.delete(oldIndOrdering, posUnL, axis = 0)
            
            # Switch label from 0 to new label
            unlabelled[posUnL, labelColumn] = labelled[posL,labelColumn] 
            
            # move newly labelled point to labelled dataframe
            labelled = np.concatenate((labelled, unlabelled[posUnL, :].reshape(1,unlabelled.shape[1])), axis=0)
            # drop from unlabelled data frame
            unlabelled = np.delete(unlabelled, posUnL, 0)
            
            # output the distance and id of the newly labelled point
            outID.append(posUnL)
            closeID.append(posL)
            
            
        # Throw error if loose or duplicate points
        if labelled.shape[0] + unlabelled.shape[0] != self.data_set.shape[0] : 
            raise Exception(
                '''The number of labelled ({}) and unlabelled ({}) points does not sum to the total ({})'''.format( len(labelled), len(unlabelled),self.data_set.shape[0]))
        # Reodered index for consistancy
        newIndex = oldIndex[outID]
        if first_run:
            self.indOrdering = indOrdering
        
        #labelled = labelled[np.argsort(indOrdering), :]
        
        # ID of point label was spread from
        closest = np.concatenate((np.array(newIndex).reshape((-1, 1)), np.array(closeID).reshape((-1, 1))), axis=1)      

        # Add new labels
        newLabels = labelled[:,self.data_set.shape[1]-1]
        #self.data_set[:,self.data_set.shape[1]-1] = newLabels
        self.data_set = labelled[np.argsort(indOrdering)]
        # invert the permutation and then assign the labels
        self.labels = self.data_set[:, -1].copy()
        return closest
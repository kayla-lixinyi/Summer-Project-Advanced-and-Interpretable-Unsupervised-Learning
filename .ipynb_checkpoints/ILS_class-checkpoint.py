import numpy as np
import pandas as pd
from scipy.signal import find_peaks_cwt
from scipy.ndimage.filters import gaussian_filter1d
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import distance_metrics

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
    def __init__(self, n_clusters = None, min_cluster_size = None, metric = 'euclidean'):

        self.n_clusters = n_clusters # need to calculate defaults based on data set input
        self.min_cluster_size = min_cluster_size
        self.metric = metric

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

        if self.min_cluster_size is None: # added just so that it will run, but need to decide on better default
            self.min_cluster_size = int (0.1 * X.shape[0])

        self.data_set = np.concatenate((np.array(X), np.zeros((X.shape[0],1))), axis = 1)
        self.rmin = []

        self.data_set[0, X.shape[1]] = 1 #initialise first label
        unlabelled = [i + 1 for i in range(X.shape[0] - 1)] # step 1

        label_spreading = self.label_spreading([0], unlabelled)

        new_centers, new_unlabelled = self.find_initial_points() # step 2

        label_spreading = self.label_spreading(new_centers, new_unlabelled, first_run = False) #step 3

        return self

    def find_minima(self):
        '''
        Written by Amanda Parker
        Given the index of the peaks used for partitioning the dataset into cluster find the point of maximum density
        OUTPUT:
            index = list index of r_min plot
        '''
        if self.rmin == []:
            raise Exception("ILS has not been run yet")

        # smooth curve
        filtered = gaussian_filter1d(self.rmin, self.min_cluster_size)
        index = np.arange(len(filtered))

        # find peaks, remove peaks and the beginning and end if implied cluster size is too small
        maxima = find_peaks_cwt(filtered, widths = len(filtered) * [self.min_cluster_size])
        maxima = [i for i in maxima if i < len(filtered) - self.min_cluster_size]
        maxima = [i for i in maxima if i > self.min_cluster_size] #removing peaks at the beginning and end

        # partition plot into cluster suggestions
        betweenMax = np.split(filtered, maxima)
        betweenIndex = np.split(index, maxima)

        # find minimum distance within plot
        minVal = [min(i) for i in betweenMax]
        subMinIndex = [np.argmin(i) for i in betweenMax]

        minima = [betweenIndex[i][subMinIndex[i]] for i in range(len(subMinIndex))]

        return minima

    def find_initial_points(self):
        '''
        Finds the points of highest density within the clusters found in the initial run and labels them in seperate classes.
        OUTPUTS:
            labelled_points: indices of labelled points (points of highest density)
                list of integers
            unlabelled_points: indices of ulabelled points (points of highest density)
                list of integers
        '''

        # get points of maximum density
        labelled_points = self.find_minima()

        counter = 1

        # label them in the data_set
        for i in labelled_points:
            self.data_set[i, -1] = counter
            counter += 1

        # unlabelled points are just the compliment
        unlabelled_points = [i for i in range(self.data_set.shape[0]) if not i in labelled_points]

        # check that we haven't missed any points
        if len(unlabelled_points) + len(labelled_points) != self.data_set.shape[0]:
            raise Exception("The number of labelled (0) and unlabelled (0) points does not sum to the total (100) in find_initial_points")

        return labelled_points, unlabelled_points

    def label_spreading(self, labelled_points, unlabelled_points, first_run = True):
        '''
        Written by Amanda Parker
        Applies iterative label spreading to the given points
        INPUTS:
            labelled_points = initial points that are already labelled
                2D array with last column the points label
            unlabelled_points = points in the data set that are not labelled
                2D array with last column the points label (0)
        OUTPUTS:
            r_min = an 1D array which contains the R_min distance for eeach iteration
        '''
        featureColumns = self.data_set[0, :self.data_set.shape[1]] # Keep original index columns in DF
        indices = np.arange(self.data_set.shape[0])
        oldIndex = np.arange(self.data_set.shape[0])

        labelled = self.data_set[labelled_points]
        unlabelled = self.data_set[unlabelled_points]

        labelColumn = self.data_set.shape[1]-1
        # lists for ordered output data
        outD = []
        outID = []
        closeID = []

        # Continue to label data points until all data points are labelled
        while len(unlabelled) > 0 :
            # Calculate labelled to unlabelled distances matrix (D)
            D = pairwise_distances(
                labelled[:, :-1],
                unlabelled[:, :-1], metric=self.metric)
            # Find the minimum distance between a labelled and unlabelled point
            # first the argument in the D matrix
            (posL, posUnL) = np.unravel_index(D.argmin(), D.shape)
            if first_run:
                self.rmin.append(D.min())

            # Switch label from 0 to new label
            unlabelled[posUnL, labelColumn] = labelled[posL,labelColumn]
            # move newly labelled point to labelled dataframe
            labelled = np.concatenate((labelled, unlabelled[posUnL, :].reshape(1,unlabelled.shape[1])), axis=0)

            # drop from unlabelled data frame
            unlabelled = np.delete(unlabelled, posUnL, 0)

            # output the distance and id of the newly labelled point
            outD.append(D.min())
            outID.append(posUnL)
            closeID.append(posL)


        # Throw error if loose or duplicate points
        if labelled.shape[0] + unlabelled.shape[0] != self.data_set.shape[0] :
            raise Exception(
                '''The number of labelled ({}) and unlabelled ({}) points does not sum to the total ({})'''.format( len(labelled), len(unlabelled),self.data_set.shape[0]) )
        # Reodered index for consistancy
        newIndex = oldIndex[outID]
        # Column 1 = indices of point in orginal dataset, Column 2 = corresponding Rmin
        orderLabelled = np.concatenate((np.array(newIndex).reshape((-1, 1)), np.array(outD).reshape((-1, 1))), axis=1)

        # ID of point label was spread from
        closest = np.concatenate((np.array(newIndex).reshape((-1, 1)), np.array(closeID).reshape((-1, 1))), axis=1)

        newLabels = labelled[:,self.data_set.shape[1]-1]
        # return
        return newLabels, np.concatenate((orderLabelled, closest), axis=1)

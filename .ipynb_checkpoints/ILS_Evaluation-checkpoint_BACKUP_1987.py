import numpy as np
import matplotlib.pyplot as plt
from ILS_class import ILS
import pandas as pd 

class ILS_Evaluation():
    
     """ Iterative Label Spreading Cluster Evaluation

    Parameters
    ----------
    
    data_set : numpy array or pandas DataFrame, 
        The last column must be the cluster labels of the point

    min_cluster_size : int, default=a third of the smallest cluster given, greater then or equal to 10
        The minimum number of data points to be considered as a cluster.
        
    significance : float, default=2.56, 
        the number of standard deviations a point must exceed the mean on 
        either side to be considered a segmentation point

    metric : String, default='euclidean'
        The valid metric for pairwise_distance.
        Must be a metric listed in pairwise.PAIRWISE_DISTANCE_FUNCTIONS or
        an option allowed by scipy.spatial.distance.pdist

    Attributes
    ----------
        
    clusters : List of numpy arrays, 
        each element in the list is a 2D array of points

    Examples
<<<<<<< HEAD
    ---------
=======
    --------
>>>>>>> refs/remotes/origin/main
    #generate clusters
    a = np.random.uniform(0, 1, (200, 2))
    b = np.random.uniform(2, 3, (200, 2))
    c = np.random.uniform(4, 5, (200, 2))
    
    # add labels
    a = np.concatenate((a, np.ones((200, 1))), axis = 1)
    b = np.concatenate((b, 2 * np.ones((200, 1))), axis = 1)
    c = np.concatenate((c, 3 * np.ones((200, 1))), axis = 1)
    
    data_set = np.concatenate((a, b, c), axis = 0)
    
    ils_eval = ILS_Evaluation(data_set, min_cluster_size = 100)
    ils_eval.plot_cluster_rmin()
    ils_eval.find_within_clusters()
<<<<<<< HEAD
=======
    
    # now try with incorrect clustering
    # generate clusters
    a = np.random.uniform(0, 1, (200, 2))
    b = np.random.uniform(2, 3, (200, 2))
    c = np.random.uniform(4, 5, (200, 2))
    
    # add labels
    a = np.concatenate((a, np.ones((200, 1))), axis = 1)
    b = np.concatenate((b, np.ones((200, 1))), axis = 1)
    c = np.concatenate((c, 2 * np.ones((200, 1))), axis = 1)
    
    data_set = np.concatenate((a, b, c), axis = 0)
    
    ils_eval = ILS_Evaluation(data_set, min_cluster_size = 100)
    ils_eval.plot_cluster_rmin()
    ils_eval.find_within_clusters()
>>>>>>> refs/remotes/origin/main

    """
    
    def __init__(self, data_set, min_cluster_size=None, significance=2.56, metric='euclidean'):
        
        if isinstance(data_set, pd.DataFrame):
            name = data_set.columns[-1]
            labels = data_set[name].unique()
            clusters = []
            for i in range(labels.shape[0]):
                clusters.append(np.array(data_set.loc[data_set[name] == labels[i]])[:, :-1])
        else:
            labels = np.unique(data_set[:, -1])
            clusters = []
            for i in range(labels.shape[0]):
                clusters.append(data_set[data_set[:, -1] == labels[i]][:, :-1])
                
        if isinstance(significance, list):
            self.significance = significance
            self.variable_sig = True
        elif isinstance(significance, float) or isinstance(significance, int):
            self.significance = significance
            self.variable_sig = False
        else:
            raise Exception("Invalid Data type for significance")
            
        self.metric = metric
        self.labels = labels
        self.clusters = clusters
        self.data_set = np.array(data_set)
        
        if min_cluster_size is None:
            cluster_sizes = [len(i) for i in self.clusters]
            self.min_cluster_size = max(np.min(cluster_sizes)//3, 10)
        else:
            self.min_cluster_size = min_cluster_size
        
    def plot_cluster_rmin(self):

        #plot the rainbow coloured rmin plot for each given cluster
        
        for i in range(len(self.clusters)):
            if self.clusters[i].shape[0] < self.min_cluster_size:
                print("Class {}: is too small to contain multiple clusters".format(self.labels[i]))
                continue
            else:
                print("Class {}".format(self.labels[i]))
            tempILS = ILS(min_cluster_size = self.min_cluster_size, metric = self.metric)
                
            tempILS.data_set = np.concatenate((self.clusters[i], np.zeros((self.clusters[i].shape[0],1))), axis = 1)
            
            tempILS.rmin = []

            tempILS.data_set[0, self.clusters[i].shape[1]] = 1 
            unlabelled = [i + 1 for i in range(self.clusters[i].shape[0] - 1)] 

            label_spreading = tempILS.label_spreading([0], unlabelled)

            tempILS.rainbow_rmin()
            
    def find_within_clusters(self, plot_labels = False):
        
        # find peaks within the given clusters
        
        n_clusters = []
        
        for i in range(len(self.clusters)):
            if self.clusters[i].shape[0] < self.min_cluster_size:
                print("Class {} is too small to contain multiple clusters".format(self.labels[i]))
                continue
            else:
                print("Class {}:".format(self.labels[i]))
            if self.variable_sig:
                tempILS = ILS(min_cluster_size = self.min_cluster_size, significance = self.significance[i], metric = self.metric)
            else:
                tempILS = ILS(min_cluster_size = self.min_cluster_size, significance = self.significance, metric = self.metric)
                
            tempILS.data_set = np.concatenate((self.clusters[i], np.zeros((self.clusters[i].shape[0],1))), axis = 1)
            
            tempILS.rmin = []

            tempILS.data_set[0, self.clusters[i].shape[1]] = 1 
            unlabelled = [i + 1 for i in range(self.clusters[i].shape[0] - 1)] 

            label_spreading = tempILS.label_spreading([0], unlabelled)
            
            new_centers, new_unlabelled = tempILS.find_initial_points()
            
            clusters_found = np.unique(tempILS.data_set[:, -1]).shape[0]
            
            n_clusters.append(clusters_found)
            
            label_spreading = tempILS.label_spreading(new_centers, new_unlabelled, first_run = False)
            
            if plot_labels == True:
                print("Class {} suggested clustering".format(self.labels[i]))
                tempILS.plot_labels()
            
        return n_clusters    
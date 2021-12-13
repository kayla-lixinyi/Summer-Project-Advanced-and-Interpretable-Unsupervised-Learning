# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 15:11:57 2021

@author: katyl
"""

from sklearn.datasets import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


import unittest

plt.style.use('ggplot')

euclidean_distance = lambda data, point: np.sqrt(np.sum(np.power(data - point, 2), axis = 1).reshape((len(data), 1)))

target = genfromtxt('target.csv', delimiter=',')
np.nan_to_num(target)


class Test(unittest.TestCase):

    # @unittest.skip("no")
    def testBlobs(self):
        no_of_clusters = 4
        
        # Create the dataset
        X, y = make_blobs(n_samples = 500, centers= 4, n_features=2,random_state=185)
        
        # Run the clustering algorithm
        ils = ILS(n_clusters=4, min_cluster_size = 50, metric = 'euclidean', plot_rmin = False, sensitivity = 0.4)
        ils.fit(X)

        # Plotting
        ils.plot_labels()
        
    
    
    # @unittest.skip("no")
    def testCircles(self):
        no_of_clusters = 2
        
        # Create the dataset
        X, y = make_circles(n_samples=500, shuffle = True, noise = 0.05, factor = 0.5, random_state = 10)
        
        # Run the clustering algorithm
        ils = ILS(n_clusters=2, min_cluster_size = 50, metric = 'euclidean', plot_rmin = False, sensitivity = 0.4)
        print(type(X))
        ils.fit(X)
        
        # Plotting
        ils.plot_labels()
        
        
        
    # @unittest.skip("no")
    def testMoons(self):
        no_of_clusters = 2
        
        # Create the dataset
        X, y = make_moons(n_samples=300, shuffle = True, noise = 0.1, random_state = 10)
        
        # Run the clustering algorithm
        ils = ILS(n_clusters=2, min_cluster_size = 100, metric = 'euclidean', plot_rmin = True, sensitivity = 0.4)
        ils.fit(X)
        
        # Plotting
        ils.plot_labels()
        
    # @unittest.skip("no")
    def testArtSet_one(self):
        no_of_clusters = 4
        
        # Create the dataset
        X, y = make_blobs(n_samples = 500, centers= 4, n_features=2,random_state=185)
        
        # Run the clustering algorithm
        ils = ILS(n_clusters=4, min_cluster_size = 50, metric = 'euclidean', plot_rmin = False, sensitivity = 0.4)
        ils.fit(target)

        # Plotting
        ils.plot_labels()
    
      


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()

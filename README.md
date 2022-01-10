# Summer-Project-Advanced-and-Interpretable-Unsupervised-Learning
## Background
The ILS algorithm was created based on the general definition of a cluster and the quality of a clustering result. ILS offers users a way to optimize hyper-parameters, such as number of clusters. Although using ILS to perform clustering was not the intended purpose, its clustering performance is considerably more successful compare to some popular mehtods. ILS is an ideal method for guiding feature selection and use in maaterials informatics. 

Why ILS? ILS does not require the input of the number of clusters or a threshold separation of any points from the outset. ILS is consistently able to identify the correct number, size, and type of clusters, regardless of the complexity of the distribution of points(including the null case and chain problem)
## Python Code Example
```
    from ILS_class import ILS
    import numpy as np
    from sklearn.datasets import * 
    X, y = make_blobs(n_samples = 500, centers= 4, n_features=2,random_state=185)   # create dataset
    ils = ILS(n_clusters=4, min_cluster_size = 50)    # initialise ILS object
    ils.fit(X)    # run ILS algorithm
    ils.plot_labels()    # plots the fitted dataset X, different colors indicates different clusters (only applicable on datasets with 2 features)
    ils.coloured_rmin()    # plots the RMin plot 
    print(ils.labels)    # cluster labels of each sample point
```

## How does ILS for clustering work
- Step 1: initialization
	Initialize one labeled point and applying ILS to obtain the ordered minimum distance Rmin(i) plot
- Step 2: cluster extraction
	The number of clusters can then be automatically extracted by identifying the peaks (due to density drops between clusters) to divide the plot into n regions
- Step 3: interative relabling
	One point relabeled in each region (preferably at the minima) to run ILS again to obtain a fully labeled data set with n clusters defined
  
*Note: The most subjective step is separating the peaks from the noise; ___add current peak finding method over here___*

## Tradeoffs
The current weakness of ILS is the scaling with number of points (as opposed to number of dimensions). Since the ILS algorithm runs the iterative label spreading method twice (first run to generate labels and second run to check labeling results from the first run), the size of the dataset would affect the scaling of the algorithm.

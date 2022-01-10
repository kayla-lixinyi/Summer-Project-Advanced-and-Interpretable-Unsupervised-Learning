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
    
## Manual Segmentation

If the user wants to specify segmentation manually then they can use the manual segmentation function.

1. View the rmin plot to identify regions to segment 

```
    ils = ILS().initial_spread(X)
    ils.plot_rmin()
```
2. Once the user has seen the plot he passes in the indexs (x1, x2, x3, ...) this will split the desired segments
```
    ils.manual_segmentation([x1, x2, x3])
    ils.plot_labels()
```

To view examples of this see Testing/ILS_tests_plots.ipynb notebook

## Semi-supervised Learning, Label Spreading

If the user already has some labelled points then they can perform the spreading once with `ILS.label_sprd_semi_sup(labelled_points, unlabelled_points)`.

An example is shown in Testing/ils_tests_plots.ipynb

## Clustering Performance/Trouble Shooting
 - Identifying quality of clustering.
 
 Firstly given a suggested clustering the user may have, it is better to use the ILS_Evaluation Class to check the quality of a given clustering. See ILS_Evaluation Section.
 
 The performance of the clustering can be identified from the colouring of distance plot where the colour corresponds to which cluster it belongs to. A good clustering result will see a small amount of colour mixing. 
 
 Good clustering
 
![Good clustering](/ReadMe_Images/Correct_Segmentation.png)

 Poor Clustering
 
![Poor clustering](/ReadMe_Images/Incorrect_Segmentation.png)

![Poor clustering](/ReadMe_Images/Incorrect_Segmentation2.png)

 - Large differences in cluster size
 
 When the cluster size of a small cluster is significantly smaller than another cluster, approximately one tenth the size or less, the segmentation method may detect multiple clusters within large clusters.
 
 If you believe the cluster are well seperated increase the significance level to around 3.4, `ILS_object = ILS(significance = 3.4)`
        
 Or specify your desired segmentation, this is shown below.
         
 - Low density cluster connected to high density cluster
 
 In these cases the segmentation method may have correctly segmented the distance plot but the spreading has not performed well. (Need to add another git repository)
 
 *Note: The most subjective step is separating the peaks from the noise; ___add current peak finding method over here___*

## Tradeoffs
The current weakness of ILS is the scaling with number of points (as opposed to number of dimensions). Since the ILS algorithm runs the iterative label spreading method twice (first run to generate labels and second run to check labeling results from the first run), the size of the dataset would affect the scaling of the algorithm.


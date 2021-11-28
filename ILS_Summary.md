## Method

1.  A Priori Insight into Cluster Characteristics
- The ILS algorithm gives a one dimensional (1D) representation of density variations in the data set
- Peaks indicate low-density jumps between clusters and troughs indicate the highest density areas.
	
*Note: Were there no scatter plots of the original data available, the information contained in the ordered Rmin(i) plots could be used to guide a clustering algorithm by identifying suitable hyperparameters or to assess the result of a clustering algorithm.*

2. ILS for clustering
- Step 1: initialization
	Initialize one labeled point and applying ILS to obtain the ordered minimum distance Rmin(i) plot
- Step 2: cluster extraction
	The number of clusters can then be automatically extracted by identifying the peaks (due to density drops between clusters) to divide the plot into n regions
- Step 3: interative relabling
	one point relabeled in each region (preferably at the minima) to run ILS again to obtain a fully labeled data set with n clusters defined

*Note: The most subjective step is separating the peaks from the noise; we chose to automate this step using a continuous wavelet transform peak finding algorithm with smoothing over p
points, which essentially sets the minimum cluster size identify clusters of smaller than p. p could easily be optimized by iteratively running ILS until the cluster specific Rmin(i) plots confirm they are single clusters.*

## Evaluation

1. Defining a Successful Clustering Result
	A) The result is well-defined and provides a clear delineation of regions which are insensitive to small changes in input parameters. 
	B) There is some relationship between the resultant clusters and the target properties. 
	C) The assigned clusters provide new information.
2. Measuring Success of a Clutering Result
	Unsuccessful Results may inculde but not  limited to:
	A) misplaced colors in clusters in the scatter plots, and misplaced colors in regions of the ILS Rmin(i)
	B) no mixing of colors in the final (low density) peak at the farright of the Rmin(i) plots
	C) no assigning a cluster of their own even though they are entirely disconnected
	
## Challenges
1. Challenges of Unsupervised Learning
- Determine the number of clusters
- Cluster outliers
- Cluster non-spherical, overlapping data
2. Different Clustering Algorithms
- Center-based 
- Nearest Neighbour
- Similarity-based
3. Chain Problems
- ILS can directly identify chains of p oints with sequentially lab elling. 
	Figure S6(c)  highlights all chains of longer than 10 data p oints. If any chain is removed, excepting the green chain that spans the two clusters, ILS will rep ort the same clustering result. If the long green chain is removed two clusters will be reported rather than one. This gives a clear test of whether any chain is considered problematic and identies when the chain problem actually exists.
# Summer-Project-Advanced-and-Interpretable-Unsupervised-Learning

## ILS Evaluation Class Use

Given a clustering this class provides tools to identify clustering mistakes. 

## Clustering Performance/Trouble Shooting
 - Identifying quality of clustering.
 
 Firstly given a suggested clustering the user may have, it is better to use the ILS_Evaluation Class to check the quality of a given clustering. See ILS_Evaluation Section.
 
 The performance of the clustering can be identified from the colouring of distance plot where the colour corresponds to which cluster it belongs to. A good clustering result will see a small amount of colour mixing. 
 
 Good clustering
 
![Good clustering](ReadMe Images/Correct Segmentation.png)

 Poor Clustering
 
![Poor clustering](ReadMe Images/Incorrect Segmentation.png)

![Poor clustering](ReadMe Images/Incorrect Segmentation2.png)

 - Large differences in cluster size
 
 When the cluster size of a small cluster is significantly smaller than another cluster, approximately one tenth the size or less, the segmentation method may detect multiple clusters within large clusters.
 
 If you believe the cluster are well seperated increase the significance level to around 3.4, `ILS_object = ILS(significance = 3.4)`
        
 Or specify your desired segmentation, this is shown below.
         
 - Low density cluster connected to high density cluster
 
 In these cases the segmentation method may have correctly segmented the distance plot but the spreading has not performed well. (Need to add another git repository)


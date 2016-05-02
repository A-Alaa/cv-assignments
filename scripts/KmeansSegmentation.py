import cv2
import numpy as np


def KmeansSegmentation( image , nclusters , threshold = 0 ,
                        maxIterations = 10 ) :
    assert (image.shape[2] == 3) , "TODO: grayscale not supported yet!"

    imageCopy = np.array(image , copy = True)
    featureVector = imageCopy.reshape(-1 , 3)

    # Column vector of the label assigned to each feature.
    labels = np.empty((featureVector.shape[0] , 1))

    # Randomly sample from the feature vector initial points as centroids.
    centroids = featureVector[np.random.choice(featureVector.shape[0] ,
                                               nclusters ,
                                               replace = False)]

    # Distance array ( featuresCount X CentroidsCount )
    distance = np.empty((featureVector.shape[0] , nclusters))

    for i in range(maxIterations) :
        for centroidIdx in range(centroids.shape[0]) :
            centroid = centroids[centroidIdx]

            # Calculate eucledian distance between feature vector and each
            # centroid.
            distance[: , centroidIdx] = np.linalg.norm( featureVector - centroid ,
                axis = 1)

        labels = np.argmin( distance , axis = 1 )
        for centroidIdx in range(centroids.shape[0]) :

            cluster = featureVector[ labels == centroidIdx ]
            newCentroid = np.mean( cluster , axis = 0 )
            centroids[ centroidIdx ] = newCentroid

    return labels, centroids
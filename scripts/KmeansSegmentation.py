import numpy as np


def KmeansSegmentation( image , nclusters , experiments = 3 , threshold = 0 ,
                        maxIterations = 4 ) :
    assert (image.shape[ 2 ] == 3) , "TODO: grayscale not supported yet!"

    imageCopy = np.array(image , copy = True , dtype = np.float)
    featureVector = imageCopy.reshape(-1 , 3  )

    # For each experiment, estimate the cluster labels and the corresponding
    # within-class sum of squares (wss).
    labels = \
        np.zeros((featureVector.shape[ 0 ] , experiments) ,
                 dtype = np.dtype(int))
    wss = \
        np.zeros(experiments)

    # Distance array ( featuresCount X CentroidsCount )
    distance = np.zeros((featureVector.shape[ 0 ] , nclusters))

    for experiment in range(experiments) :
        # Randomly sample from the feature vector initial points as centroids.
        centroids = \
            featureVector[ np.random.choice(featureVector.shape[ 0 ] ,
                                            nclusters , replace = False ) ]

        for iteration in range(maxIterations) :
            # Calculate eucledian dist( points , each centroid).
            for centroidIdx in range(centroids.shape[ 0 ]) :
                distance[ : , centroidIdx ] = \
                    np.linalg.norm(featureVector - centroids[ centroidIdx ] ,
                                   axis = 1 )

            labels[ : , experiment ] = np.argmin(distance , axis = 1 )

            for centroidIdx in range(centroids.shape[ 0 ]) :
                cluster = featureVector[
                    labels[ : , experiment ] == centroidIdx ]
                newCentroid = np.mean(cluster , axis = 0)
                wss[ experiment ] += \
                    np.sum(np.linalg.norm(cluster - newCentroid ,
                                          axis = 1 , keepdims = True ))

                centroids[ centroidIdx ] = newCentroid

    finalLabels = labels[ : , np.argmin(wss) ]
    finalCentroids = __getCentroids__(featureVector , nclusters , finalLabels)
    return finalLabels , finalCentroids , np.min(wss)


def __getCentroids__( population , nclusters , labels ) :
    centroids = np.empty((nclusters , population.shape[ 1 ]))

    for centroidIdx in range(nclusters) :
        cluster = population[ labels == centroidIdx ]
        centroid = np.mean(cluster , axis = 0)
        centroids[ centroidIdx ] = centroid

    return centroids

import numpy as np
from collections import deque

class AgglomerativeClustering :
    def __init__( self , featureVector , nclusters = 6 ) :

        featureCopy = np.array(featureVector , copy = True , dtype = np.float)
        self.features = featureCopy.reshape(-1)

        self.sortingIndices = np.argsort(self.features)
        self.recoverIndices = np.argsort(self.sortingIndices)

        self.root = np.array(range(self.features.shape[ 0 ]))
        self.nclusters = nclusters
        self.clusterId = self.features.shape[ 0 ]
        self.children = {}
        self.sortedFeatures = \
            np.array(self.features[ self.sortingIndices ] , copy = True)

        print "Root shape:%d"%self.root.shape

        self.__mergeClusters__()
        self.__reconstructSegments__()
        self.labels = \
            np.array(self.root[ self.recoverIndices ] , copy = True)

    def __reconstructSegments__( self ) :
        print "Reconstruction.."
        clusters = deque([ self.clusterId - 1 ])
        while clusters and len(clusters) < self.nclusters :
            supercluster = clusters.popleft()
            try:
                (child1 , child2) = self.children.get(int(supercluster))
                clusters.extend([ int(child1) , int(child2) ])
            except:
                pass
        print clusters

        [ self.__markSegment__(root) for root in clusters ]
        print "[DONE] Reconstruction.."

    def __markSegment__( self , clusterRoot ) :
        segment = [ clusterRoot ]

        while segment :
            root = segment.pop()
            if root < self.features.shape[ 0 ] :
                self.root[ root ] = clusterRoot
            else :
                (child1 , child2) = self.children.get(root)
                segment.extend([ int(child1) , int(child2) ])

    def __mergeClusters__( self ) :
        clustersCount = self.features.shape[ 0 ]
        clusters = self.sortedFeatures
        root = np.array(self.root , copy = True)
        while clustersCount > 1 :

            if clustersCount != 2 :
                diff = np.abs(clusters - np.roll(clusters , +1))
                minIndices = np.argwhere(diff == diff.min())
            else :
                minIndices = np.array([ 0 ] , dtype = int)

            for i in range(minIndices.shape[ 0 ]) :
                idx = minIndices[ i ]
                cluster1 = root[ idx ]
                cluster2 = root[ (idx + 1) % root.shape[ 0 ] ]

                assert cluster1 != cluster2 , \
                    "Algorithm error, clusters:%d" % root.shape[ 0 ]

                clusterId = self.clusterId
                self.clusterId += 1

                self.children.update({clusterId: (cluster1 , cluster2)})

                root[ root == cluster1 ] = clusterId
                root[ root == cluster2 ] = clusterId

            # Reduce clusters
            unique , uniqueIndices = np.unique(root , return_index = True)
            for i in range(unique.shape[0]) :
                clusters[ clusters == unique[ i ]] = \
                    np.mean( clusters[ clusters == unique[ i ] ] )
            clusters = clusters[ uniqueIndices ]

            root = root[ uniqueIndices ]
            clustersCount = root.shape[ 0 ]
            print clustersCount

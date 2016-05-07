import numpy as np


class AgglomerativeClustering :
    def __init__( self , image , nclusters = 6 ) :
        imageCopy = np.array(image , copy = True , dtype = np.float)
        self.features = imageCopy.reshape(-1)
        self.sortingIndices = np.argsort(self.features)
        self.recoverIndices = np.argsort(self.sortingIndices)
        self.root = np.array(range(self.features.shape[ 0 ]))
        self.nclusters = nclusters
        self.clusterId = self.features.shape[ 0 ]
        self.children = {}
        self.sortedFeatures = \
            np.array( self.features[ self.sortingIndices] , copy = True )

        self.__mergeClusters__()
        self.__reconstructSegments__()
        self.features = self.sortedFeatures[ self.recoverIndices ]
        self.image = np.reshape( self.features , image.shape )

    def __reconstructSegments__(self):
        print "Reconstruction.."
        clusters = [ self.clusterId - 1 ]
        while len(clusters) < self.nclusters :
            supercluster = clusters.pop()
            (child1,child2) = self.children.get( int(supercluster) )
            clusters.extend( [int(child1),int(child2)] )


        [ self.__markSegment__( root ) for root in clusters ]
        print "[DONE] Reconstruction.."

    def __markSegment__(self,clusterRoot ):
        segment = [ clusterRoot ]

        while segment :
            root = segment.pop()
            if root < self.features.shape[0] :
                self.root[ root ] = clusterRoot
            else :
                (child1,child2) = self.children.get( root )
                segment.extend([ int(child1) , int(child2)])

        self.sortedFeatures[ self.root == clusterRoot ] = \
            np.mean( self.sortedFeatures[ self.root == clusterRoot ] )

    def __mergeClusters__( self ) :
        clustersCount = self.features.shape[ 0 ]
        clusters = self.features[ self.sortingIndices ]
        root = np.array( self.root , copy = True )
        while clustersCount > 1 :
            diff = np.abs(clusters - np.roll(clusters , -1))
            minDist = diff.min()
            minIndices = np.argwhere(diff == diff.min())

            for i in range(minIndices.shape[ 0 ]) :
                clusterId = self.clusterId
                self.clusterId += 1
                idx = minIndices[ i ]
                cluster1 = root[ idx ]
                cluster2 = root[ (idx + 1) % root.shape[ 0 ] ]

                self.children.update({self.clusterId:(cluster1,cluster2)})

                root[ root == cluster1 ] = clusterId
                root[ root == cluster2 ] = clusterId

            # Reduce clusters
            unique , uniqueIndices = np.unique(root , return_index = True)
            clusters = clusters[ uniqueIndices ]
            root = root[ uniqueIndices ]
            clustersCount = root.shape[ 0 ]
            print clustersCount
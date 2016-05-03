import numpy as np

class RegionGrowingSegmentation :
    def __init__( self , grayScaleImage , threshold = 180 ) :
        assert (grayScaleImage.dtype == np.dtype('uint8')) , \
            "Input image must be gray-scale."
        self.image = np.array(grayScaleImage , copy = True)
        self.__regionCounter__ = 0
        self.labelImage = np.full(grayScaleImage.shape , -1 , dtype = int)
        self.threshold = 10

    def newRegion( self , seedPoint ) :
        regionId = self.__regionCounter__
        self.__regionCounter__ += 1

        try :
            self.labelImage[ seedPoint[ 0 ] , seedPoint[ 1 ] ]
        except :
            assert True , "Error: invalid position."

        if self.labelImage[ seedPoint[ 0 ] , seedPoint[ 1 ] ] == -1 :
            self.labelImage[ seedPoint[ 0 ] , seedPoint[ 1 ] ] = regionId
        else :
            print "Already segmented region!"
            return self.labelImage[ seedPoint[ 0 ] , seedPoint[ 1 ] ]

        visitors = self.__getFourNeighbors__(seedPoint)
        regionMean = int( self.image[ seedPoint[ 0 ] , seedPoint[ 1 ] ])

        while visitors :
            row , col  = visitors.pop()
            if np.abs( regionMean - self.image[ row , col ] ) < self.threshold :
                self.labelImage[ row , col ] = regionId
                visitors.extend(self.__getFourNeighbors__((row , col)))
                regionMean = \
                    np.mean(self.image[ self.labelImage == regionId ])

        self.image[ self.labelImage == regionId ] = regionMean
        print "[DONE] New Segmentation"
        return regionId

    def __getFourNeighbors__( self , seedPoint ) :
        delta = [ (1 , 0) , (0 , 1) , (-1 , 0) , (0 , -1) ]
        neighbors = [ ]
        for d in delta :
            neighbor = np.asarray( seedPoint ) + d
            try :
                if self.labelImage[ neighbor[ 0 ] , neighbor[ 1 ] ] == -1 :
                    neighbors.append(neighbor)
            except :
                continue
        return neighbors

    def __getEightNeighbors__( self , seedPoint ) :
        delta = [ (1 , 0) , (1 , 1) , (0 , 1) , (-1 , 1) , (-1 , 0) ,
                  (-1 , -1) , (0 , -1) , (1 , -1) ]
        neighbors = [ ]
        for d in delta :
            neighbor = tuple(sum(p) for p in zip(d , seedPoint))
            try :
                if self.labelImage[ neighbor[ 0 ] , neighbor[ 1 ] ] == -1 :
                    neighbors.append(neighbor)
            except :
                continue
        return neighbors
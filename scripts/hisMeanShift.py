import numpy as np

"""
An implementation to the mean-shift segmentation on LUV image
"""

class meanShiftSeg:

    def __init__(self, image, windowSize):
        # self.image = np.asarray( image, copy = True )
        self.image = image
        assert (self.image.shape[2] == 3), "The Image must be of three channels LUV "
        self.windowSize = windowSize*4
        self.segmentedImage = image
        self.colorSpace = np.zeros( (256,256) )
        self.numofClusters = 265/self.windowSize
        self.clustersUV = np.zeros( shape=(self.numofClusters, 2) )


    def __makeColorDataSpace__(self):
        compU = np.ndarray.flatten( self.image[:,:,1] )
        compV = np.ndarray.flatten( self.image[:,:,2] )
        self.colorSpace[ compU,compV ] = 1

    def applyMeanShift(self):
        self.__makeColorDataSpace__()
        wSize = self.windowSize
        # halfWSize = self.windowSize/2
        numOfWindPerDim = np.floor(np.sqrt( self.numofClusters ))
        clustersTemp = []
        for itrRow in range( numOfWindPerDim ):
            for itrCol in range( numOfWindPerDim ):
                cntrRow, cntrCol = self.__windowIterator__( itrRow*wSize,itrCol*wSize )
                clustersTemp.append( (cntrRow, cntrCol) )

        self.clustersUV = np.array( clustersTemp )

        self.__classifyColors__()

        return self.segmentedImage


    def __classifyColors__(self):
        wSize = self.windowSize
        numOfWindPerDim = np.floor(np.sqrt( self.numofClusters ))
        for row in range( self.image.shape[0] ):
            for col in range( self.image.shape[1] ):
                pixelU = self.segmentedImage[row,col,1]
                pixelV = self.segmentedImage[row,col,2]
                windowIdx = np.floor( pixelV/wSize ) + (numOfWindPerDim)*np.floor( pixelU/wSize )
                self.segmentedImage[row,col,1] = self.clustersUV[windowIdx, 0]
                self.segmentedImage[row,col,2] = self.clustersUV[windowIdx, 1]



    def __windowIterator__(self, row, col):
        wSize = self.windowSize
        hWSize = wSize/2
        prevRow = 0
        prevCol = 0
        window = self.colorSpace[ row:row+wSize,col:col+wSize ]
        newRow, newCol = self.__findCntrMass__( window )
        while( prevRow != newRow-hWSize and prevCol != newCol-hWSize ):
            prevRow = newCol-hWSize
            prevCol = newCol-hWSize
            window = self.colorSpace[ prevRow:prevRow+wSize,prevCol:prevCol+wSize ]
            newRow, newCol = self.__findCntrMass__( window )

        return row + newRow, col + newCol

    def __findCntrMass__(self, window):
        """
        Calculate the window's center of mass
        :param window:
        :return:
        """
        momntIdx = range( self.windowSize )
        nonZeroIdx = np.transpose(np.nonzero( window ))

        totalMass = np.max(np.cumsum( window ))
        #Moment around column #0 ( around the x-axis )
        momentCol = np.max(np.cumsum(window.cumsum( axis=0 )[self.windowSize-1]*momntIdx))
        cntrCol = np.round(1.0*momentCol/totalMass)

        #Moment around row #0 ( around the y-axis )
        momentRow = np.max(np.cumsum(window.cumsum( axis=1 )[:,self.windowSize-1]*momntIdx))
        cntrRow = np.round(1.0*momentRow/totalMass)

        return cntrRow, cntrCol
    def __findEclidDist__(self, row, col):
        """
        Find the the Euclidean distance for the pixel from its position ( row, col )
        :param row:
        :param col:
        :return:
        """
        dist = np.sqrt( (row**2 + col**2 ))
        dist = np.round( dist )
        return dist

    def getSegmentedImage(self):
        return self.segmentedImage

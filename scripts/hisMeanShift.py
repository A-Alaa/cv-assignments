import numpy as np
from matplotlib import pyplot as plt

"""
An implementation to the mean-shift segmentation on LUV image
"""

class meanShiftSeg:

    def __init__(self, image, windowSize):
        self.image = np.array( image, copy = True )
        assert (self.image.shape[2] == 3), "The Image must be of three channels LUV "
        self.windowSize = 2**windowSize
        print "Window size : " , self.windowSize
        self.segmentedImage = np.array( image, copy = True )
        ## The LUV is 256X3 , so the color space to be clustered is 256X256
        self.colorSpace = np.zeros( (256,256) )
        self.numofClusters = np.int(256/self.windowSize)**2
        print "# of clusters : ", self.numofClusters
        self.clustersUV = np.zeros( shape=(self.numofClusters, 2) )
        # print self.numofClusters, self.windowSize,256/self.windowSize


    def __makeColorDataSpace__(self):
        """
        This function populate the color-space to be clustered
        :return:
        """

        # compU = np.ndarray.flatten( self.image[:,:,1] )
        # compV = np.ndarray.flatten( self.image[:,:,2] )
        compU = np.reshape( self.image[:,:,1], (-1,1) )
        compV = np.reshape( self.image[:,:,2], (-1,1) )
        compUV = np.transpose(np.array((compU[:,0],compV[:,0])))
        print compU.shape, compV.shape, compUV.shape
        # self.colorSpace[ compU,compV ] = 1
        for u,v in compUV :
                # print (u, v)
                self.colorSpace[ u,v ] += 1

        # print self.segmentedImage
        # print compV.max(), compU.max()
        # plt.plot( compV.tolist(), compU.tolist(),'bo' )
        # plt.show()
        # exit(1)
    def applyMeanShift(self):
        """
        Apply the mean-shift to the color-space, then classify the image U-V components
        :return: segmented image
        """
        # return self.image
        print "Apply shift .... "
        self.__makeColorDataSpace__()
        wSize = self.windowSize
        # halfWSize = self.windowSize/2
        numOfWindPerDim = np.int(np.sqrt( self.numofClusters ))
        clustersTemp = []
        for itrRow in range( numOfWindPerDim ):
            for itrCol in range( numOfWindPerDim ):
                cntrRow, cntrCol = self.__windowIterator__( itrRow*wSize,itrCol*wSize )
                # print itrRow*wSize,itrCol*wSize
                clustersTemp.append( (cntrRow, cntrCol) )

        self.clustersUV = np.array( clustersTemp )
        print " Clusters formed "
        self.__classifyColors__()

        return self.segmentedImage




    def __windowIterator__(self, row, col):
        """
        This function iterate in the given window indices, to find its center of mass
        :param row:
        :param col:
        :return:
        """
        # print " Iterrating to find mean value"
        wSize = self.windowSize
        hWSize = wSize/2
        prevRow = 0
        prevCol = 0
        # print row,":",row+wSize,col,":",col+wSize
        window = self.colorSpace[ row:row+wSize,col:col+wSize ]
        # print window.shape
        newRow, newCol = self.__findCntrMass__( window )
        numOfIter = 0
        while( prevRow != newRow-hWSize and prevCol != newCol-hWSize ):
            if( numOfIter > np.sqrt(self.numofClusters) ):
                break
            # print prevRow, prevCol
            # print newRow, newCol
            prevRow = newCol-hWSize
            prevCol = newCol-hWSize
            # print numOfIter
            # print prevRow+row,":",prevRow+row+wSize," ", prevCol+col,":", prevCol+col+wSize
            nxtRow = (prevRow+row)%(256-wSize)
            nxtCol = (prevCol+col)%(256-wSize)
            window = self.colorSpace[ nxtRow:nxtRow+wSize,nxtCol:nxtCol+wSize ]
            newRow, newCol = self.__findCntrMass__( window )
            numOfIter += 1
        return row + newRow, col + newCol

    def __classifyColors__(self):
            """
            This function classify the image component based on the its value, which is the index in the color-space
            see also : https://spin.atomicobject.com/2015/05/26/mean-shift-clustering/
            to understand what is colo-space
            :return:
            """
            wSize = self.windowSize
            numOfWindPerDim = np.int(np.sqrt( self.numofClusters ))
            for row in range( self.image.shape[0] ):
                for col in range( self.image.shape[1] ):
                    pixelU = self.segmentedImage[row,col,1]
                    pixelV = self.segmentedImage[row,col,2]
                    windowIdx = np.int( np.int(pixelV/wSize)  + np.int(numOfWindPerDim*( pixelU/wSize )))
                    # print pixelV/wSize ,windowIdx, numOfWindPerDim, self.numofClusters, self.windowSize
                    # print self.clustersUV.shape
                    # exit(1)
                    # print windowIdx
                    self.segmentedImage[row,col,1] = self.clustersUV[windowIdx, 1]
                    self.segmentedImage[row,col,2] = self.clustersUV[windowIdx, 0]
                    self.segmentedImage[row,col,0] = self.image[row,col,0]



    def __findCntrMass__(self, window):
        """
        Calculate the window's center of mass
        :param window:
        :return:
        """
        # print window.shape
        # exit(130)
        # assert window.shape[0] == 0, 'Window of empty'
        momntIdx = range( self.windowSize )
        nonZeroIdx = np.transpose(np.nonzero( window ))
        totalMass = np.max(np.cumsum( window ))
        if (totalMass == 0):
            return self.windowSize/2 , self.windowSize/2
        if ( totalMass > 0 ):
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

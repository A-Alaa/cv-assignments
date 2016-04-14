import cv2
import numpy as np
from scipy.signal import convolve2d
import scipy.ndimage.filters as filters


class HarrisCorner:
    def __init__(self, sigma, image):
        self.sigma = sigma
        self.image = np.matrix(image, copy=True, dtype=np.float32)
        self.supressedCorners = []
        self.gradientX = []
        self.gradientY = []
        self.cornerIndex = []
        self.cornerIndexsupressed = []
        self.responseMat = np.zeros(image.shape)
        self.cornersImage = np.zeros(image.shape)

    def __findGradients__(self):
        # Calulate the Gradient of the image in X,Y directions
        # gradientHKernel = np.array([ [ -1 , 0 , 1 ] ,
        #                              [ -1 , 0 , 1 ] ,
        #                              [ -1 , 0 , 1 ] ])

        gradientHKernel = np.array([[-1, 0, 1],
                                    [-2, 0, 2],
                                    [-1, 0, 1]])
        self.gradientX = convolve2d(self.image, gradientHKernel, mode='same')
        self.gradientY = convolve2d(self.image, gradientHKernel.transpose(), mode='same')

    def __calcHarrisMat__(self):
        self.__findGradients__()
        # M = SUM( 3X3 window )
        # H = det(M) - k*(trace(M))*(trace(M))
        # k = 0.04 <=> 0.06 , we will assume it is 0.05
        # det(M) = (Ix*Ix)*(Iy*Iy) - (Ix*Iy)*(Ix*Iy) ,  trace(M) = (Ix*Ix) + (Iy*Iy)
        gaussianKernel1D = cv2.getGaussianKernel(3, self.sigma)
        window = gaussianKernel1D * gaussianKernel1D.transpose()
        # window = np.array([ [ 1 , 1 , 1 ] ,
        #                     [ 1 , 1 , 1 ] ,
        #                     [ 1 , 1 , 1 ] ])

        gradXSquared = self.gradientX * self.gradientX
        gradYSquared = self.gradientY * self.gradientY
        gradXgradY = self.gradientX * self.gradientY

        # Calculate the summation of the window's value ( IxIx, IyIy, IxIy)

        mIxIx = convolve2d(gradXSquared, window, mode='same')
        mIyIy = convolve2d(gradYSquared, window, mode='same')
        mIxIy = convolve2d(gradXgradY, window, mode='same')

        # Calculate the |M|
        detOfMatrix = (mIxIx * mIyIy) - (mIxIy * mIxIy)
        # Calculate the trace()
        traceOfMatrix = mIxIx + mIyIy

        self.responseMat = detOfMatrix - 0.05 * traceOfMatrix * traceOfMatrix
        # self.responseMat = detOfMatrix / ( traceOfMatrix * traceOfMatrix )

    def findCorners(self, threshold):
        self.__calcHarrisMat__()
        # self.__localMaxima__()
        corners = []
        for row in range(self.responseMat.shape[0]):
            for col in range(self.responseMat.shape[1]):
                if self.responseMat[row][col] >= threshold:
                    corners.append((row, col))
                    self.cornersImage[row][col] = self.responseMat[row, col]
        # cv2.imshow("shit", self.cornersImage)
        # cv2.waitKey(0)
        # for row in range( self.responseMat.shape[0] ):
        #     for col in range( self.responseMat.shape[1] ):
        #         if self.responseMat[row][col] > (threshold/((threshold+1)*(threshold+1))):
        #             corners.append(( row, col ))

        self.cornerIndex = np.array(corners)
        # return self.cornerIndex
    def localMaxima(self, windowSize):
        # Find the local Max in the window windowSize
        imageRow = self.cornersImage.shape[0]
        imageCol = self.cornersImage.shape[1]
        halfWin = int( windowSize/2 )
        corners = []
        for (row, col) in self.cornerIndex:
            if (row > halfWin+1 and row < (imageRow - halfWin-1) and col > halfWin+1 and col < (imageCol -halfWin-1) ):
                imageSec = self.cornersImage[ (row-halfWin) : (row+halfWin) , (col-halfWin) : (col+halfWin) ]
                imageSec = np.matrix( imageSec, copy= True )
                maxIdx = imageSec.argmax()
                rMax, cMax = np.unravel_index( maxIdx, imageSec.shape )
                # print rMax, cMax
                rMax = row + rMax - halfWin
                cMax = col + cMax - halfWin
                corners.append((rMax, cMax))
                # print imageSec
        self.cornerIndexsupressed = np.array(corners)

        # Remove Redundant Corners, that appear in local Maximum suppression
        nonRedundCorners = []
        for itr in range( 1, self.cornerIndexsupressed.shape[0] ):
            currRow = self.cornerIndexsupressed[itr, 0]
            currCol = self.cornerIndexsupressed[itr, 1]
            if( nonRedundCorners.count(( currRow, currCol )) == 0 ):
                nonRedundCorners.append(( currRow, currCol ))
        self.cornerIndexsupressed = np.array( nonRedundCorners )
        return self.cornerIndexsupressed

def getResponseMat(self):
    return self.responseMat

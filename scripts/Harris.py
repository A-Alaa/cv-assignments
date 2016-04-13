import cv2
import numpy as np
from scipy.signal import convolve2d


class HarrisCorner:
    def __init__(self, sigma, image):
        self.sigma = sigma
        self.image = np.matrix(image, copy=True, dtype=np.float32)
        self.gradientX = []
        self.gradientY = []
        self.cornerIndex = []
        self.responseMat = []
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
        self.__localMaxima__()
        corners = []
        for row in range(self.responseMat.shape[0]):
            for col in range(self.responseMat.shape[1]):
                if self.responseMat[row][col] >= threshold:
                    corners.append((row, col))
                    self.cornersImage[row][col] = self.responseMat[row][col]

        # for row in range( self.responseMat.shape[0] ):
        #     for col in range( self.responseMat.shape[1] ):
        #         if self.responseMat[row][col] > (threshold/((threshold+1)*(threshold+1))):
        #             corners.append(( row, col ))

        self.cornerIndex = np.array(corners)
        return self.cornerIndex

    def __localMaxima__(self):
        # Find the local Max in the window
        for col in range(self.responseMat.shape[0]):
            for row in range(self.responseMat.shape[1]):
                try:
                    maxNeighbor = max([[self.responseMat[row - 1, col - 1],
                                        self.responseMat[row - 1, col],
                                        self.responseMat[row - 1, col + 1],
                                        self.responseMat[row, col - 1],
                                        self.responseMat[row, col + 1],
                                        self.responseMat[row + 1, col - 1],
                                        self.responseMat[row + 1, col],
                                        self.responseMat[row + 1, col + 1]]])
                    if( maxNeighbor > self.responseMat[ row , col ]) :
                        self.responseMat[ row , col ] = 0 ;
                except:
                    pass


def getResponseMat(self):
    return self.responseMat

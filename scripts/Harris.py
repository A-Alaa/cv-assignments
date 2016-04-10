import cv2
import numpy as np
from scipy.signal import convolve2d

class HarrisCorner :
    def __init__(self, sigma, image):
        self.sigma = sigma
        self.image = np.matrix(image, copy=True, dtype=np.float32)
        self.gradientX = []
        self.gradientY = []
        self.cornerIndex = []
        self.responseMat = []

    def __findGradients__(self):
        # Calulate the Gradient of the image in X,Y directions
        # gradientHKernel = np.array([ [ -1 , 0 , 1 ] ,
        #                              [ -1 , 0 , 1 ] ,
        #                              [ -1 , 0 , 1 ] ])

        gradientHKernel = np.array([ [ -1 , 0 , 1 ] ,
                                     [ -2 , 0 , 2 ] ,
                                     [ -1 , 0 , 1 ] ])
        self.gradientX = convolve2d( self.image, gradientHKernel, mode='same' )
        self.gradientY = convolve2d( self.image, gradientHKernel.transpose(), mode='same' )


    def __calcHarrisMat__(self):
        self.__findGradients__()

        # H = det(M) - k*(trace(M))*(trace(M))
        # k = 0.04 <=> 0.06 , we will assume it is 0.05
        # det(M) = (Ix*Ix)*(Iy*Iy) - (Ix*Iy)*(Ix*Iy) ,  trace(M) = (Ix*Ix) + (Iy*Iy)

        detOfMatrix = ( self.gradientX * self.gradientX ) * ( self.gradientY * self.gradientY ) - \
            ( self.gradientX * self.gradientY ) * ( self.gradientX * self.gradientX )

        traceOfMatrix = ( self.gradientX * self.gradientX ) + ( self.gradientY * self.gradientY )

        self.responseMat = detOfMatrix - 0.05 * traceOfMatrix * traceOfMatrix


    def findCorners(self, thresold):
        self.__calcHarrisMat__()

        corners = []
        for row in range( self.responseMat.shape[0] ):
            for col in range( self.responseMat.shape[1] ):
                if self.responseMat[row][col] >= thresold:
                    corners.append(( row, col ))

        self.cornerIndex = np.array( corners )
        return self.cornerIndex


    def getResponseMat(self):
        return self.responseMat
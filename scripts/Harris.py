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
        self.responseMat = []
        self.cornersImage = np.zeros(image.shape,np.uint8)

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

        M_IxIx = convolve2d(gradXSquared, window, mode='same')
        M_IyIy = convolve2d(gradYSquared, window, mode='same')
        M_IxIy = convolve2d(gradXgradY, window, mode='same')

        # Calculate the |M|
        detOfMatrix = (M_IxIx * M_IyIy) - (M_IxIy * M_IxIy)
        # Calculate the trace()
        traceOfMatrix = M_IxIx + M_IyIy

        self.responseMat = detOfMatrix - 0.05 * traceOfMatrix * traceOfMatrix
        # self.responseMat = detOfMatrix / ( traceOfMatrix * traceOfMatrix )

    def findCorners(self, threshold):
        self.__calcHarrisMat__()

        corners = []
        for row in range(self.responseMat.shape[0]):
            for col in range(self.responseMat.shape[1]):
                if self.responseMat[row][col] >= threshold:
                    corners.append((row, col))
                    self.cornersImage[row][col] = 255
        cv2.imshow("shit", self.cornersImage)
        # cv2.waitKey(0)
        # for row in range( self.responseMat.shape[0] ):
        #     for col in range( self.responseMat.shape[1] ):
        #         if self.responseMat[row][col] > (threshold/((threshold+1)*(threshold+1))):
        #             corners.append(( row, col ))

        self.cornerIndex = np.array(corners)
        print self.cornerIndex.shape
        return self.cornerIndex

    def localMaxima(self, windowSize=7):
        # Find the local Max in the window windowSize
        # cornersMax = filters.minimum_filter( self.cornersImage, 1 )
        # self.supressedCorners = (self.cornersImage==cornersMax)
        # cornersMax = filters.minimum_filter( self.cornersImage, 1 )
        # self.supressedCorners = (self.cornersImage==cornersMax)
        kernel = np.ones((3,3),np.uint8)

        self.supressedCorners = cv2.erode( self.cornersImage, kernel, iterations=2 )
        cv2.imshow("shit2", self.supressedCorners)
        corners = []
        for row in range(self.supressedCorners.shape[0]):
            for col in range(self.supressedCorners.shape[1]):
                if self.supressedCorners[row][col] !=0 :
                    corners.append((row, col))

        self.cornerIndexsupressed = np.array(corners)

        return self.cornerIndexsupressed

    def getResponseMat(self):
        return self.responseMat

import numpy as np
from scipy.signal import convolve2d
import cv2


class mySIFT :
    def __init__( self , image , corners  ) :
        self.image = image
        self.corners = corners
        self.magnitude = []
        self.phaseImage = []
        self.__calculateGradients__()
        self.featres = []
    def __calculateGradients__( self ):
        sobelHKernel = np.array([[-1, 0, 1],
                                 [-2, 0, 2],
                                 [-1, 0, 1]])

        gradientX = convolve2d( self.image, sobelHKernel, mode='same')
        gradientY = convolve2d( self.image, sobelHKernel.transpose(), mode='same')
        self.magnitude = np.sqrt( np.square(gradientX) + np.square(gradientY))

        phaseImage = np.arctan2( gradientY, gradientX) * (180.0 / np.pi)
        self.phaseImage = (10 * np.round(phaseImage / 10.0)) + 180 ;
        print self.magnitude.shape
        print self.phaseImage.shape

    def __featureDescription__(self):

        for corner in self.corners :
            row = corner[0]
            col = corner[1]

            # Check if boundaries are exceeded.
            if( row + 7 >= self.image.shape[0]  or
                row - 8 <= 0 or
                col + 7 >= self.image.shape[1]  or
                col - 8 <= 0 ) :
                continue


            featureMagnitude = self.magnitude[(row-8):(row+8) ,(col-8):(col+8) ]
            featureMagnitude = np.matrix( featureMagnitude , copy = True )


            featurePhase = self.phaseImage[ (row-8):(row+8) , (col-8):(col+8) ]
            featurePhase =  np.matrix( featurePhase , copy = True )

            # get the most  frequent  orientation
            frequency = np.zeros(37)
            for index in range(37):
                angle = index * 10                                   #[10,20,30,40,........,360]
                for i in range(featurePhase.shape[0]):
                        for j in range(featurePhase.shape[1]):
                            if featurePhase[i,j] == angle :
                                frequency[index] += 1
            print frequency, np.max(frequency)


            gaussianKernel1D = cv2.getGaussianKernel( 16 , 1.5 )
            gaussianKernel2D = gaussianKernel1D * gaussianKernel1D.transpose()
            print np.max(gaussianKernel2D)
            featureMagnitude *=gaussianKernel2D



    def getSIFTDescriptors(self):
        self.__featureDescription__()

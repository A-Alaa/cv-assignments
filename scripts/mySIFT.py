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
        # divide the angle histogram to 36 bins
        self.phaseImage = ((10 * np.round(phaseImage / 10.0)) + 180) % 360
        # print self.magnitude.shape
        # print self.phaseImage.shape

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

            # get the main orientation
            angleFrequency = np.zeros(37)
            dominantAngle = 0
            gradients=np.zeros(37)
            for index in range(37):
                angle = index * 10                                   #[0 10,20,30,40,........,350]
                for i in range(featurePhase.shape[0]):
                    for j in range(featurePhase.shape[1]):
                        if featurePhase[i,j] == angle:
                            angleFrequency[index] += 1
                            gradients[index] += featureMagnitude[i,j]

            # print angleFrequency
            # print np.max(angleFrequency)
            # dominantAngle=np.argmax(angleFrequency)*10
            # print dominantAngle

            # print gradients
            featureMagnitudeMax = np.max(gradients)
            dominantAngle = np.argmax(gradients) * 10
            # print dominantAngle
            # print featureMagnitudeMax

            # subtract the dominant angle from feature phase
            featurePhase = featurePhase - dominantAngle
            # print featurePhase
            # Down sampling the 36-bin to 8-bin
            featurePhase8Bin = ((45 * np.round(self.phaseImage / 45.0)) )%360
            print featurePhase8Bin
            gaussianKernel1D = cv2.getGaussianKernel( 16 , 1.5 )
            gaussianKernel2D = gaussianKernel1D * gaussianKernel1D.transpose()
            # print np.max(gaussianKernel2D)
            featureMagnitude *=gaussianKernel2D



    def getSIFTDescriptors(self):
        self.__featureDescription__()

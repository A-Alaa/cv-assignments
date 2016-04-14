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

            gaussianKernel1D = cv2.getGaussianKernel(16, 1.5)
            gaussianKernel2D = gaussianKernel1D * gaussianKernel1D.transpose()

            featureMagnitudeWeighted = featureMagnitude * gaussianKernel2D

            featurePhase = self.phaseImage[ (row-8):(row+8) , (col-8):(col+8) ]
            featurePhase =  np.matrix( featurePhase , copy = True )

            # get the main orientation
            angleFrequency = np.zeros(36)
            dominantAngle = 0
            gradients=np.zeros(36)
            for index in range(36):
                angle = index * 10                                   #[0 10,20,30,40,........,350]
                for i in range(featurePhase.shape[0]):
                    for j in range(featurePhase.shape[1]):
                        if featurePhase[i,j] == angle:
                            angleFrequency[index] += 1
                            gradients[index] += featureMagnitudeWeighted[i,j]

            featureMagnitudeMax = np.max(gradients)
            dominantAngle = np.argmax(gradients) * 10
            print "dominat Angle" , dominantAngle

            # subtract the dominant angle from feature phase
            featurePhaseAdjusted = featurePhase - dominantAngle

            # Down sampling the 36-bin to 8-bin
            featurePhase8Bin = ((45 * np.round(featurePhaseAdjusted / 45.0)) ) % 360
            print featurePhase8Bin.shape
            print featureMagnitudeWeighted.shape

            # get gradients magnitude for each 4*4 region
            # test first quad
            featurePhaseQuad = featurePhase8Bin[0:4 , 0:4]
            featureMagnitudeWeightedQuad=featureMagnitudeWeighted[0:4 , 0:4]
            print featureMagnitudeWeightedQuad
            print featurePhaseQuad


            featureGradients = np.zeros(8)
            for index in range(8):
                angle = index * 45  # [0  45 90 135 ... 315]
                for i in range(featurePhaseQuad.shape[0]):
                    for j in range(featurePhaseQuad.shape[1]):
                        if featurePhaseQuad[i, j] == angle:
                            gradients[index] += featureMagnitudeWeighted[i, j]

            print featureGradients

    def getSIFTDescriptors(self):
        self.__featureDescription__()

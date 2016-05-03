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
        self.features = []

    def __calculateGradients__( self ):
        # 1- Get gradient magnitude and  gradient phase  of image
        sobelHKernel = np.array([[-1, 0, 1],
                                 [-2, 0, 2],
                                 [-1, 0, 1]])

        gradientX = convolve2d( self.image, sobelHKernel, mode='same')
        gradientY = convolve2d( self.image, sobelHKernel.transpose(), mode='same')
        self.magnitude = np.sqrt( np.square(gradientX) + np.square(gradientY))

        phaseImage = np.arctan2( gradientY, gradientX) * (180.0 / np.pi)

        # 2-Sampling  the angle histogram to 36 bins
        self.phaseImage = ((10 * np.round(phaseImage / 10.0)) + 180) % 360

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

            gaussianKernel2DNormalized=gaussianKernel2D / np.amax(gaussianKernel2D)

            #3- Weight the gradients magnitude by 16x16 gaussian kernel

            featureMagnitudeWeighted = featureMagnitude * gaussianKernel2DNormalized

            featurePhase = self.phaseImage[ (row-8):(row+8) , (col-8):(col+8) ]
            featurePhase =  np.matrix( featurePhase , copy = True )

            # 4- Get the main orientation of the patch at max gradient magnitude
            gradientMagnitudes = self.__getGradientsMagnitudes__(featurePhase,featureMagnitudeWeighted,36)

            dominantAngle = np.argmax(gradientMagnitudes) * 10 # Get index of angle so we multiply by 10 to get the corresponding angle
            print "Dominant Angle for feature at corner ", corner, "=", dominantAngle

            #5- Subtract the dominant angle from feature phase
            featurePhaseAdjusted = featurePhase - dominantAngle
            #6- Down sampling the 36-bin to 8-bin
            featurePhase8Bin = ((45 * np.round(featurePhaseAdjusted / 45.0)) ) % 360

            #7- Get gradient for angle histogram (8 bins)
            featureDesc = []
            for i in range(0, 13, 4):
                for k in range(0, 4):
                    featurePhaseQuad = featurePhase8Bin[i:i + 4, 4 * k: 4 * (k + 1)]
                    featureMagnitudeWeightedQuad = featureMagnitudeWeighted[i:i + 4, 4 * k: 4 * (k + 1)]
                    featureVector = self.__getGradientsMagnitudes__(featurePhaseQuad, featureMagnitudeWeightedQuad, 8)
                    featureDesc = featureDesc + featureVector       # the plus  operator  concatinates two vectors

            if max(featureDesc) != 0:
                 featureDesc /= max(featureDesc)
            self.features.append(featureDesc)

    # Get gradient at defined angles
    def __getGradientsMagnitudes__(self, phase, magnitude, bins):
        gradientMagnitudes1 = []
        angles = range(0, 360, 360/bins)  # [0 10 20 30,,,,,,,,350] or [0 45 90 .......315]
        for angle in angles:
            indices = np.where(phase == angle)
            gradients = np.sum(magnitude[indices])
            gradientMagnitudes1.append(gradients)
        return gradientMagnitudes1

    def getSIFTDescriptors(self):
        self.__featureDescription__()
        return self.features

import numpy as np
import cv2
import copy


def __cannySmoothing__( img , ksize , sigma ) :
    return cv2.GaussianBlur(img , (ksize , ksize) , sigma)


def __cannyEdgeDetection__( image , ksize ) :
    sobelH = cv2.Sobel(image , -1 , 1 , 0 , ksize = ksize)
    sobelV = cv2.Sobel(image , -1 , 0 , 1 , ksize = ksize)

    phaseImage = np.arctan2(sobelV , sobelH) * (180.0 / np.pi)

    # Assign phase values to nearest [ 0 , 45 , 90 ,  135 ]
    phaseImage = ((45 * np.round(phaseImage / 45.0)) + 180) % 180;

    # return gradient magnitude image, phase image.
    return np.sqrt(sobelH * sobelH + sobelV * sobelV) , phaseImage


def __cannyNonMaximumSupression__( gradientImage , phaseImage ) :
    # make a copy.
    thinEdgedImage = np.array(gradientImage , copy = True)

    for row in range(1 , gradientImage.shape[0] - 1) :
        for col in range(1 , gradientImage.shape[1] - 1) :
            # If already supressed, proceed to the next pixel.
            if thinEdgedImage[row , col] == 0 :
                continue

            theta = phaseImage[row , col]
            maxGradient = gradientImage[row , col];

            if theta == 0 :
                maxGradient = max([gradientImage[row , col - 1] ,
                                   gradientImage[row , col] ,
                                   gradientImage[row , col + 1]])

            elif theta == 45 :
                maxGradient = max([gradientImage[row - 1 , col - 1] ,
                                   gradientImage[row , col] ,
                                   gradientImage[row + 1 , col + 1]])


            elif theta == 90 :
                maxGradient = max([gradientImage[row - 1 , col] ,
                                   gradientImage[row , col] ,
                                   gradientImage[row + 1 , col]])


            elif theta == 135 :
                maxGradient = max([gradientImage[row + 1 , col - 1] ,
                                   gradientImage[row , col] ,
                                   gradientImage[row - 1 , col + 1]])
            else :
                print("Unexpected theta value")
                print theta
                exit(1)

            if gradientImage[row , col] < maxGradient :
                thinEdgedImage[row , col] = 0

    return thinEdgedImage


def __cannyDoubleThresholding__( image , minThreshold , maxThreshold ) :
    doubleThresholdImages = copy.deepcopy(image)
    image = doubleThresholdImages
    for row in range(0 , image.shape[0] - 1) :
        for col in range(0 , image.shape[1] - 1) :
            if image[row , col] < minThreshold :
                image[row , col] = 0.0
            elif image[row , col] >= maxThreshold :
                image[row , col] = 1.0

    return doubleThresholdImages


def __cannyEdgeTracking__( images , maxThreshold , minThreshold ) :
    edgeTrackingImages = copy.deepcopy(images)
    image = edgeTrackingImages
    for row in range(1 , image.shape[0] - 1) :
        for col in range(1 , image.shape[1] - 1) :

            if image[row , col] > maxThreshold :
                image[row , col] = 1.0

            elif image[row , col] < minThreshold :
                image[row , col] = 0

            else :
                maxGradient = max([[image[row - 1 , col - 1] ,
                                    image[row - 1 , col] ,
                                    image[row - 1 , col + 1] ,
                                    image[row , col - 1] ,
                                    image[row , col + 1] ,
                                    image[row + 1 , col - 1] ,
                                    image[row + 1 , col] ,
                                    image[row + 1 , col + 1]]])
                if maxGradient >= maxThreshold :
                    image[row , col] = maxThreshold
                else :
                    image[row , col] = 0

    return edgeTrackingImages


def myCanny( image , ksize , sigma , minThreshold , maxThreshold ) :
    # 1.Guassian Blurring.
    smoothedImage = __cannySmoothing__(image , ksize , sigma)

    # 2.Gradient and Phase Image based on Sobel edge detection.
    gradientImage , phaseImage = __cannyEdgeDetection__(smoothedImage , ksize)

    # 3.Non-Maximum Supression for thinning edges.
    nmsImage = __cannyNonMaximumSupression__(gradientImage , phaseImage)

    # 4.Double Thresholding.
    doubleThresholdedImage = __cannyDoubleThresholding__(nmsImage ,
                                                         minThreshold ,
                                                         maxThreshold)

    # 5.Edge Tracking by Hysterysis.
    edgeTrackedImage = __cannyEdgeTracking__(doubleThresholdedImage ,
                                             maxThreshold ,
                                             minThreshold)

    return edgeTrackedImage
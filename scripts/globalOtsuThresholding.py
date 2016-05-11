# threshold gray image using otsu thresholding method

import cv2 as cv
import numpy as np
import math
import copy

# otsuThresholding Takes an original image 
# to be thresholded and threshold it 
# by otsu thresholding method
# returns otsu threshold , thresholded image 

def globalOtsuThresholding(image):
    imageSize =  np.size(image)
    # 1 - Calculate Histogram of image
    grayLevels = range(0,256)
    histogram = [0] * 256
    for level in grayLevels:
        histogram[level] = len(np.extract(np.asarray(image) == grayLevels[level], image))

    # 2 - Get between class variance for each gray Level (threshold)
    betweenClassVariance = []
    for level in grayLevels:
        threshold = level
        backgroundGrayLevels = np.extract(np.asarray(grayLevels) < threshold, grayLevels)
        foregroundGrayLevels = np.extract(np.asarray(grayLevels) >= threshold, grayLevels)
        backgroundHist = []
        foregroundHist = []

        backgroundWeight = 0
        foregroundWeight = 0
        backgroundMean =   0
        foregroundMean =   0

        # get corresponding histogram for each region [ background , foreground]
        if len(backgroundGrayLevels):
            for level in backgroundGrayLevels:
                backgroundHist.append(histogram[level])
            # calculate weight of background
                backgroundWeight = float(sum(backgroundHist)) / imageSize
            # calculate  mean of background if background exists
            if backgroundWeight:
                backgroundMean = np.sum(np.multiply(backgroundGrayLevels, np.asarray(backgroundHist))) / float(sum(backgroundHist))

        if len(foregroundGrayLevels):
            for level in foregroundGrayLevels:
                foregroundHist.append(histogram[level])
            # calculate weight of foreground
                foregroundWeight = float(sum(foregroundHist)) / imageSize
            # calculate  mean of foreground if foreground exists
            if foregroundWeight:
                foregroundMean = np.sum(np.multiply(foregroundGrayLevels, np.asarray(foregroundHist))) / float(sum(foregroundHist))
        # get between class variance at current gray level
        betweenClassVariance.append(backgroundWeight * foregroundWeight * (backgroundMean - foregroundMean) * (backgroundMean - foregroundMean))

    # 3 - Get maximum gray level corresponding to maximum betweenClassVariance
    maxbetweenClassVariance = np.max(betweenClassVariance)
    otsuThreshold = betweenClassVariance.index(maxbetweenClassVariance)
    outputImage = copy.deepcopy(image)

    # convert to binary
    for r in range(0, image.shape[0]):
        for c in range(0, image.shape[1]):
            if image[r, c] >= otsuThreshold:
                outputImage[r, c] = 255
            else:
                outputImage[r, c] = 0

    return  outputImage, otsuThreshold

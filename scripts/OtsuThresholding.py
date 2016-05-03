# threshold gray image using otsu thresholding method

import cv2 as cv
import numpy as np
import math
import copy

# otsuThresholding Takes an original image
# to be thresholded and threshold it using
# otsu thresholding method
# returns otsu threshold , thresholded image
def otsuThresholding(image):
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
        wb, wf, meanb, meanf = 0, 0, 0, 0
        # get corresponding histogram for each region [ background , foreground]
        if len(backgroundGrayLevels):
            for level in backgroundGrayLevels:
                backgroundHist.append(histogram[level])
            # calculate weight of backgound
            wb = float(sum(backgroundHist)) / imageSize
            if wb:  meanb = np.sum(np.multiply(backgroundGrayLevels, np.asarray(backgroundHist))) / float(sum(backgroundHist))

        if len(foregroundGrayLevels):
            for level in foregroundGrayLevels:
                foregroundHist.append(histogram[level])
            # calculate weight of foreground
            wf = float(sum(foregroundHist)) / imageSize
            if wf: meanf = np.sum(np.multiply(foregroundGrayLevels, np.asarray(foregroundHist))) / float(sum(foregroundHist))
        # get between class variance at current lelvel
        betweenClassVariance.append(wb * wf * math.pow(meanb - meanf, 2))

    # 3 - Get maximum gray level corresponding to maximum betweenClassVariance
    otsuThreshold = betweenClassVariance.index(max(betweenClassVariance))

    bwImage = copy.deepcopy(image)
    # convert the image to black and white image
    for r in range(0, image.shape[0]):
        for c in range(0, image.shape[1]):
            if image[r, c] > otsuThreshold:
                bwImage[r, c] = 255
            else:
                bwImage[r, c] = 0

    return  bwImage, otsuThreshold

# threshold gray image using otsu thresholding method

import cv2 as cv
import numpy as np
import math
import copy

# otsuThresholding Takes an original image or block
# to be thresholded and threshold it using
# otsu thresholding method
# returns otsu threshold , thresholded image or thresholded
# block

def otsu_thresholding(input):
    inputSize =  np.size(input)
    # 1 - Calculate Histogram of image
    grayLevels = range(0,256)
    histogram = [0] * 256
    for level in grayLevels:
        histogram[level] = len(np.extract(np.asarray(input) == grayLevels[level], input))

    # 2 - Get between class variance for each gray Level (threshold)
    betweenClassVariance = []
    for level in grayLevels:
        threshold = level
        backgroundGrayLevels = np.extract(np.asarray(grayLevels) < threshold, grayLevels)
        foregroundGrayLevels = np.extract(np.asarray(grayLevels) >= threshold, grayLevels)
        backgroundHist = []
        foregroundHist = []
        wb = 0     # wb background weight
        wf = 0     # wf foreground weight
        meanb = 0  # meanb background mean
        meanf = 0  # meanf foreground mean

        # get corresponding histogram for each region [ background , foreground]
        if len(backgroundGrayLevels):
            for level in backgroundGrayLevels:
                backgroundHist.append(histogram[level])
            # calculate weight of backgound
            wb = float(sum(backgroundHist)) / inputSize
            # calculate  mean of background if background exists
            if wb:
                meanb = np.sum(np.multiply(backgroundGrayLevels, np.asarray(backgroundHist))) / float(sum(backgroundHist))

        if len(foregroundGrayLevels):
            for level in foregroundGrayLevels:
                foregroundHist.append(histogram[level])
            # calculate weight of foreground
            wf = float(sum(foregroundHist)) / inputSize
            # calculate  mean of foreground if foreground exists
            if wf:
                meanf = np.sum(np.multiply(foregroundGrayLevels, np.asarray(foregroundHist))) / float(sum(foregroundHist))
        # get between class variance at current gray  level
        betweenClassVariance.append(wb * wf * (meanb - meanf) * (meanb - meanf))

    # 3 - Get maximum gray level corresponding to maximum betweenClassVariance
    maxbetweenClassVariance = np.max(betweenClassVariance)
    maxCorrespondingLevels = np.where(np.asarray(betweenClassVariance) == maxbetweenClassVariance)
    otsu_threshold = np.max(maxCorrespondingLevels)
    binary_output = copy.deepcopy(input)

    # convert to binary
    for r in range(0, input.shape[0]):
        for c in range(0, input.shape[1]):
            if input[r, c] >= otsu_threshold:
                binary_output[r, c] = 255
            else:
                binary_output[r, c] = 0

    return  binary_output, otsu_threshold

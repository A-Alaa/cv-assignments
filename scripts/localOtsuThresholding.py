
import cv2 as cv
import numpy as np
import math
import copy

# localOtsuThresholding Takes an original image ,blockSize
# to be thresholded and threshold it
# by otsu thresholding method into blocks
# returns thresholded image

def localOtsuThresholding(image,block_size):

    # resize image to square image [ max dimension] because block size is square!
    if image.shape[0] != image.shape[1]:
        if image.shape[0] > image.shape[1]:
            resizedImage = cv.resize(image, (image.shape[0], image.shape[0]))
        else:
            resizedImage = cv.resize(image, (image.shape[1], image.shape[1]))
    else:
        resizedImage =image
    rows = resizedImage.shape[0]
    cols = resizedImage.shape[1]

    if block_size > image.shape[0] and block_size > image.shape[1]:
        print "Error local thresholding , block size should be smaller than image size!"
        exit()

    # output image
    thresholdedImage = np.zeros(resizedImage.shape)

    # ------------------------------------ otsu thresholding algorithm------------------------------------------#
    for r in range(0, rows, block_size):
        for c in range(0, cols, block_size):
            # Extarct blocks
            block = resizedImage[r:min(r + block_size, rows), c:min(c + block_size, cols)]
            blockSize = np.size(block)
            # 1 - Calculate Histogram of each block
            grayLevels = range(0, 256)
            histogram = [0] * 256
            for level in grayLevels:
                histogram[level] = len(np.extract(np.asarray(block) == grayLevels[level], block))

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
                backgroundMean = 0
                foregroundMean = 0

                # get corresponding histogram for each region [ background , foreground]
                if len(backgroundGrayLevels):
                    for level in backgroundGrayLevels:
                        backgroundHist.append(histogram[level])
                        # calculate weight of background
                        backgroundWeight = float(sum(backgroundHist)) / blockSize
                    # calculate  mean of background if background exists
                    if backgroundWeight:
                        backgroundMean = np.sum(np.multiply(backgroundGrayLevels, np.asarray(backgroundHist))) / float(
                            sum(backgroundHist))

                if len(foregroundGrayLevels):
                    for level in foregroundGrayLevels:
                        foregroundHist.append(histogram[level])
                        # calculate weight of foreground
                        foregroundWeight = float(sum(foregroundHist)) / blockSize
                    # calculate  mean of foreground if foreground exists
                    if foregroundWeight:
                        foregroundMean = np.sum(np.multiply(foregroundGrayLevels, np.asarray(foregroundHist))) / float(
                            sum(foregroundHist))
                # get between class variance at current gray level
                betweenClassVariance.append(backgroundWeight * foregroundWeight * (backgroundMean - foregroundMean) * (backgroundMean - foregroundMean))

            # 3 - Get maximum gray level corresponding to maximum betweenClassVariance
            maxbetweenClassVariance = np.max(betweenClassVariance)
            otsuThreshold = betweenClassVariance.index(maxbetweenClassVariance)

            # convert to binary [ (0 , 255) only]
            thresholdedBlock = np.zeros(block.shape)
            for row in range(0, block.shape[0]):
                for col in range(0, block.shape[1]):
                    if block[row, col] >= otsuThreshold:
                        thresholdedBlock[row, col] = 255
                    else:
                        thresholdedBlock[row, col] = 0

            # fill the output image for each block
            thresholdedImage[r:min(r + block_size, rows), c:min(c + block_size, cols)] = thresholdedBlock

    # resize output  image back to original size
    thresholdedImage = cv.resize(thresholdedImage, (image.shape[1], image.shape[0]))
    print thresholdedImage.shape
    return thresholdedImage

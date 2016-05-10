import cv2 as cv
import numpy as np
import copy

# localOptimalThresholding Takes an original image ,block_size
# to be thresholded and divide it's into blocks and threshold each
# block by optimal thresholding method
# returns thresholded image

def localOptimalThresholding(image, block_size):
    if image.shape[0] != image.shape[1]:
        if image.shape[0] > image.shape[1]:
            resizedImage = cv.resize(image, (image.shape[0], image.shape[0]))
        else:
            resizedImage = cv.resize(image, (image.shape[1], image.shape[1]))
    else:
        resizedImage = image
    rows = resizedImage.shape[0]
    cols = resizedImage.shape[1]

    if block_size <= 2:
        print "Error in local thresholding , block size should be greater than 2 ! "
        exit()

    if block_size > image.shape[0] and block_size > image.shape[1]:
        print "Error local thresholding , block size should be smaller than image size!"
        exit()

    outputImage = np.zeros(resizedImage.shape)
    #------------------------------------ optimal thresholding algorithm------------------------------------------#

    for r in range(0, rows, block_size):
        for c in range(0, cols, block_size):
            # Extarct blocks
            block = resizedImage[r:min(r + block_size,rows), c:min(c + block_size, cols)]
            # get  initial background mean  (4corners)
            background = [block[0, 0], block[0, block.shape[1]-1], block[block.shape[0]-1, 0], block[block.shape[0]-1, block.shape[1]-1]]
            background_mean = np.mean(background)
            # get  initial foreground mean
            foreground_mean = np.mean(block) - background_mean
            # get  initial threshold
            thresh = (background_mean + foreground_mean) / 2.0
            while True:
                old_thresh = thresh
                new_foreground = block[np.where(block >= thresh)]
                new_background = block[np.where(block < thresh)]
                if new_background.size:
                    new_background_mean = np.mean(new_background)
                else:
                    new_background_mean = 0
                if new_foreground.size:
                    new_foreground_mean = np.mean(new_foreground)
                else:
                    new_foreground_mean = 0
                # update threshold
                thresh = (new_background_mean + new_foreground_mean) / 2
                if old_thresh == thresh:
                    break

            # convert to binary [ (0 , 255) only]
            thresholdedBlock = np.zeros(block.shape)
            for row in range(0, block.shape[0]):
                for col in range(0, block.shape[1]):
                    if block[row, col] >= thresh:
                        thresholdedBlock[row, col] = 255
                    else:
                        thresholdedBlock[row, col] = 0

            # fill the output image for each block
            outputImage[r:min(r + block_size, rows) , c:min(c + block_size, cols)] = thresholdedBlock

    # resize output  image back to original size
    outputImage = cv.resize(outputImage, (image.shape[1], image.shape[0]))
    return outputImage



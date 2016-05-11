import cv2 as cv
import numpy as np
import copy

# globalOptimalThresholding Takes an original image
# to be thresholded and threshold it using
# optimal thresholding method
# returns optimal threshold , thresholded image


def globalOptimalThresholding(image):
    rows = image.shape[0]
    cols = image.shape[1]
    # get  initial background mean  (4corners)
    background = [image[0, 0], image[0, cols-1], image[rows-1, 0], image[rows-1, cols-1]]
    background_mean = np.mean(background)
    # get  initial foreground mean
    foreground_mean = np.mean(image) - background_mean
    # get  initial threshold
    thresh = (background_mean + foreground_mean) / 2.0

    while True:
        old_thresh = thresh
        new_foreground = image[np.where(image >= thresh)]
        new_background = image[np.where(image < thresh)]
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
    binary_output = copy.deepcopy(image)
    for r in range(0, rows):
        for c in range(0, cols):
            if image[r, c] >= thresh:
                binary_output[r, c] = 255
            else:
                binary_output[r, c] = 0
    return binary_output, round(thresh,2)


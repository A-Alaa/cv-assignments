import cv2 as cv
import numpy as np
import copy

# optimalThresholding Takes an original image or block
# to be thresholded and threshold it using
# optimal thresholding method
# returns optimal threshold , thresholded image or
# thresholded block


def optimal_thresholding(input):
    rows = input.shape[0]
    cols = input.shape[1]
    #  get  initial background mean  (4corners)
    background = [input[0, 0], input[0, cols-1], input[rows-1, 0], input[rows-1, cols-1]]
    background_mean = np.mean(background)
    #  get  initial foreground mean
    foreground_mean = np.mean(input) - background_mean
    #  get  initial threshold
    thresh = (background_mean + foreground_mean) / 2

    while True:
        old_thresh = thresh
        new_foreground = input[np.where(input >= thresh)]
        new_background = input[np.where(input < thresh)]
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
        # if round(old_thresh, 2) == round(thresh, 2):
        if old_thresh == thresh:
            break

    # convert to binary [ (0 , 255) only]
    binary_output = copy.deepcopy(input)
    for r in range(0, rows):
        for c in range(0, cols):
            if input[r, c] >= thresh:
                binary_output[r, c] = 255
            else:
                binary_output[r, c] = 0
    return binary_output, round(thresh,2)


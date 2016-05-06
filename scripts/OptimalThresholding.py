import cv2 as cv
import numpy as np
import copy

# optimalThresholding Takes an original image
# to be thresholded and threshold it using
# optimal thresholding method
# returns optimal threshold , thresholded image

def optimal_thresholding(image):
    rows = image.shape[0]
    cols = image.shape[1]
    #  Extract initial background  (4corners)
    background = [image[0, 0], image[0, cols-1], image[rows-1, 0], image[rows-1, cols-1]]
    background_mean = np.mean(background)
    # Extract initial foreground (object)
    foreground_sum = np.sum(image) - np.sum(background)
    foreground_mean = foreground_sum / (np.size(image) - len(background))
    thresh = (background_mean + foreground_mean) / 2

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

        thresh = (new_background_mean + new_foreground_mean) / 2
        if old_thresh == thresh:
            break

    binary_image = copy.deepcopy(image)
    # convert the image to black and white image
    for r in range(0, rows):
        for c in range(0, cols):
            if image[r, c] >= thresh:
                binary_image[r, c] = 255
            else:
                binary_image[r, c] = 0
    return binary_image, thresh


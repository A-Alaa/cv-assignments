import cv2 as cv
import numpy as np
import copy
import sys
sys.path.append('../scripts')
from OtsuThresholding import otsu_thresholding


def optimal_thresholding(image):
    rows = image.shape[0]
    cols = image.shape[1]
    #  Extract initial background  (4corners)
    background = [image[0, 0], image[0, cols-1], image[rows-1, 0], image[rows-1, cols-1]]
    print "background" , background
    background_mean = np.mean(background)
    # Extract initial foreground (object)
    foreground_sum = np.sum(image) - np.sum(background)
    foreground_mean = foreground_sum / (np.size(image) - len(background))
    thresh = (background_mean + foreground_mean) / 2
    print thresh

    while True:
        old_thresh = thresh
        new_foreground = image[np.where(image >= thresh)]
        new_background = image[np.where(image < thresh)]
        if(new_background.size):
            new_background_mean = np.mean(new_background)
        else:
            new_background_mean = 0
        if (new_foreground.size):
            new_foreground_mean = np.mean(new_foreground)
        else:
            new_foreground_mean = 0

        thresh = (new_background_mean + new_foreground_mean) / 2
        if old_thresh == thresh:
            break

    bwImage = copy.deepcopy(image)
    # convert the image to black and white image
    for r in range(0, rows):
        for c in range(0, cols):
            if image[r, c] >= thresh:
                bwImage[r, c] = 255
            else:
                bwImage[r, c] = 0
    return bwImage, thresh


# load gray images
imageColor = cv.imread('../images/assignment4/0019hv3.bmp')
imageGray = cv.imread('../images/assignment4/0019hv3.bmp', cv.IMREAD_GRAYSCALE)
img = imageGray
N = 10
# #
# for r in range(0, img.shape[0]):
#     for c in range(0, img.shape[1]):
#         block = imageGray[r:r+N , c:c+N]
#         print type(block)
#         print block.shape
#

print img.shape
block = imageGray[0:0+N , 0:0+N]
print block
print type(block)
print block.shape

thresholded_block , otsu_threshold = otsu_thresholding(block)
print thresholded_block
print "threshold", otsu_threshold

#
# #threshold gray image using optiaml thresholding method
# thresholded_image , optimal_threshold = optimal_thresholding(imageGray)
#
# print "optimal threshold", optimal_threshold
#
# cv.namedWindow('original image',cv.WINDOW_NORMAL)
# cv.imshow('original image', imageGray)
#
# cv.imshow('thresholded image using optimal thresholding', thresholded_image)
#
# cv.waitKey(0)
# cv.destroyAllWindows()

import cv2 as cv
import sys
import numpy as np
sys.path.append('../scripts')
from globalOtsuThresholding import globalOtsuThresholding
from localOtsuThresholding import localOtsuThresholding

# load gray images
imageColor = cv.imread('../images/assignment4/Beads.jpg')
imageGray = cv.imread('../images/assignment4/Beads.jpg', cv.IMREAD_GRAYSCALE)

# ---------------------------Global thresholding--------------------------------------- #

thresholdedImageGlobal , otsuThreshold = globalOtsuThresholding(imageGray)
print "global threshold using otsu thresholding method = ", otsuThreshold

# ---------------------------local thresholding---------------------------------------- #

blockSize = 256
thresholdedImageLocal = localOtsuThresholding(imageGray , blockSize)

# display all images
cv.imshow('Original image', imageGray)
cv.imshow('global thresholding', thresholdedImageGlobal)
window_name = "local thresholding of block size " + str(blockSize)
cv.imshow(window_name, thresholdedImageLocal)
cv.waitKey(0)
cv.destroyAllWindows()

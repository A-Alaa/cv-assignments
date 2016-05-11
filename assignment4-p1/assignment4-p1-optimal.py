import cv2 as cv
import sys
import numpy as np
sys.path.append('../scripts')

from globalOptimalThresholding import globalOptimalThresholding
from localOptimalThresholding import localOptimalThresholding

# load gray images
imageColor = cv.imread('../images/assignment4/Beads.jpg')
imageGray = cv.imread('../images/assignment4/Beads.jpg', cv.IMREAD_GRAYSCALE)

# ---------------------------Global thresholding--------------------------------------- #

thresholdedImageGlobal , optimal_threshold = globalOptimalThresholding(imageGray)
print "global threshold using optimal thresholding method = ", optimal_threshold
# ---------------------------local thresholding---------------------------------------- #

blockSize = 16
thresholdedImageLocal = localOptimalThresholding(imageGray , blockSize)

cv.imshow('Original image', imageGray)
cv.imshow('global thresholding', thresholdedImageGlobal)
cv.imshow("Local thresholding of block size " + str(blockSize) , thresholdedImageLocal)
cv.waitKey(0)
cv.destroyAllWindows()


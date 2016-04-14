import cv2
import numpy as np
import copy as cp
from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join
import numpy as np
from scipy.signal import convolve2d
import scipy.ndimage.filters as filters

import sys

sys.path.append('../scripts')

from Harris import HarrisCorner
# ____________________________________________________________________________________________________________________#
cv2.__version__
# ____________________________________________________________________________________________________________________#

imageColor = cv2.imread('../images/assignment2/Regular-Shapes.jpg')
imageGray = cv2.imread('../images/assignment2/Regular-Shapes.jpg', 0)

# imageGrayMax = filters.minimum_filter( imageGray, 1 )
#
# cv2.imshow("image", imageGrayMax)


harrisCorners = HarrisCorner( 0.5, imageGray )
harrisCorners.findCorners( 100000 )
cornerIndex = harrisCorners.localMaxima( 19)

print imageGray.shape, cornerIndex.shape
# print cornerIndex

for index in cornerIndex:
    x = index[1]
    y = index[0]
    cv2.drawMarker( imageColor, (x,y), (0,0,0), markerSize=30 )

cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.imshow('image', imageColor)

cv2.waitKey(0)
cv2.destroyAllWindows()

# dst = cv2.cornerHarris(imageGray,5,9,0.05)
# #result is dilated for marking the corners, not important
# dst = cv2.dilate(dst,None)
# # Threshold for an optimal value, it may vary depending on the image.
# imageColor[dst>0.5*dst.max()]=[0,0,255]
# cv2.imshow('dst',imageColor)
# if cv2.waitKey(0) & 0xff == 27:
#     cv2.destroyAllWindows()
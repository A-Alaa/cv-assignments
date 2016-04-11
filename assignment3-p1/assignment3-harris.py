import cv2
import numpy as np
import copy as cp
from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join
import numpy as np
import sys

sys.path.append('../scripts')

from Harris import HarrisCorner
# ____________________________________________________________________________________________________________________#
cv2.__version__
# ____________________________________________________________________________________________________________________#

imageColor = cv2.imread('../images/assignment2/Regular-Shapes.jpg')
imageGray = cv2.imread('../images/assignment2/Regular-Shapes.jpg', 0)

harrisCorners = HarrisCorner( 0.5, imageGray )
cornerIndex = harrisCorners.findCorners( 100 )

print imageGray.shape, cornerIndex.shape, harrisCorners.getResponseMat

for index in cornerIndex:
    x = index[1]
    y = index[0]
    cv2.drawMarker( imageColor, (x,y), (0,0,0), 1, 2 )

cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.imshow('image', imageColor)

cv2.waitKey(0)
cv2.destroyAllWindows()

# dst = cv2.cornerHarris(imageGray,5,9,0.04)
# #result is dilated for marking the corners, not important
# dst = cv2.dilate(dst,None)
# # Threshold for an optimal value, it may vary depending on the image.
# imageColor[dst>0.01*dst.max()]=[0,0,255]
# cv2.imshow('dst',imageColor)
# if cv2.waitKey(0) & 0xff == 27:
#     cv2.destroyAllWindows()
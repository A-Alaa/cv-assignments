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

imageFileName = '../images/assignment3/Black&WhiteCow2.jpg'

imageColor = cv2.imread(imageFileName)
imageGray = cv2.imread(imageFileName, 0)

# get current time before harris
start = cv2.getTickCount()

harrisCorners = HarrisCorner( 1, imageGray )
harrisCorners.findCorners( 100000)
cornerIndex = harrisCorners.localMaxima( 25)
# get current time after harris
end = cv2.getTickCount()

time = (end - start)/ cv2.getTickFrequency()
print "Time taken by harris" , time , "seconds"


print imageGray.shape, cornerIndex.shape
# print cornerIndex

for index in cornerIndex:
    x = index[1]
    y = index[0]
    cv2.drawMarker( imageColor, (x,y), (0,0,0), markerSize=3 )

cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.imshow('image', imageColor)

cv2.waitKey(0)
cv2.destroyAllWindows()

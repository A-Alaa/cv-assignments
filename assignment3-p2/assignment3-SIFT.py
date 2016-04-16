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
from mySIFT import mySIFT
from FeatureMatching import FeatureMatcher

imageColor = cv2.imread('../images/assignment2/Regular-Shapes.jpg')
imageGray = cv2.imread('../images/assignment2/Regular-Shapes.jpg', 0)

imageRotatedColor = cv2.imread('../images/assignment2/Regular-Shapes-Rotated.jpg')
imageRotatedGray = cv2.imread('../images/assignment2/Regular-Shapes-Rotated.jpg', 0)

# original Image

harrisCorners = HarrisCorner( 0.5, imageGray )
harrisCorners.findCorners( 100000 )
cornerIndexOrig = harrisCorners.localMaxima( 19)

print "10 corners" ,cornerIndexOrig[0:10]
print "type" , type(cornerIndexOrig)


print imageGray.shape, cornerIndexOrig.shape
# print cornerIndex

for index in cornerIndexOrig:
    x = index[1]
    y = index[0]
    cv2.drawMarker( imageColor, (x,y), (0,0,0), markerSize=20 )

cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.imshow('image', imageColor)


SIFT1 = mySIFT( imageGray , cornerIndexOrig[0:10])
# features of size no*features * feature-desc
features1 = SIFT1.getSIFTDescriptors()

print "No of features descriptors of original image  " ,len(features1) , "of size : " , len(features1[0])
# print "Feature 1 descriptor " ,features1[0]
# print "Feature 2 descriptor " ,features1[1]
# print "Feature 3 descriptor " ,features1[2]

# Rotated Image

harrisCorners = HarrisCorner( 0.5, imageRotatedGray )
harrisCorners.findCorners( 100000 )
cornerIndexRotated = harrisCorners.localMaxima( 19)

print imageRotatedGray.shape, cornerIndexRotated.shape
# print cornerIndex

for index in cornerIndexRotated:
    x = index[1]
    y = index[0]
    cv2.drawMarker( imageRotatedColor, (x,y), (0,0,0), markerSize=30 )

cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.imshow('image', imageRotatedColor)


SIFT2= mySIFT( imageRotatedGray , cornerIndexRotated[0:10] )
# features of size no*features * feature-desc [ n x 1]
features2=SIFT2.getSIFTDescriptors()

print "No of features descriptors  of Rotated Image" ,len(features2) , "of size : " , len(features2[0])
# print "Feature 1 descriptor " ,features2[0]
# print "Feature 2 descriptor " ,features2[1]
# print "Feature 3 descriptor " ,features2[2]
#
# feature Matching
myMatcher = FeatureMatcher(features1,cornerIndexOrig[0:10],features2,cornerIndexRotated[0:10])
matchingPoints=myMatcher.getMatchingPoints()
print "matching Points " , matchingPoints

# matchedFeatures = myMatcher.getMatchingFeatures()
# print "matched features",matchedFeatures

cv2.waitKey(0)
cv2.destroyAllWindows()

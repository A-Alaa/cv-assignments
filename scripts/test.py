import numpy as np
import sys
import cv2

imageColor = cv2.imread('../images/assignment3/Pyramids1.jpg')
imageGray = cv2.imread('../images/assignment3/Pyramids1.jpg', 0)
sys.path.append('../scripts')

from mySIFT import mySIFT

m = mySIFT( imageGray , [(20,20),(50,50),(70,60)] )
# features of size no*features * feature-desc
features=m.getSIFTDescriptors()
print "No of features descriptors " ,len(features) , "of size : " , len(features[0])
print "Feature 1 descriptor " ,features[0]
print "Feature 2 descriptor " ,features[1]

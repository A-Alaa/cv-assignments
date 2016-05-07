from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile , join
import numpy as np
import cv2
import sys
sys.path.append('../scripts')

from hisMeanShift import meanShiftSeg

imagePath = "../images/assignment4/seg2.jpg"

imageRGB = cv2.imread( imagePath )

imageLUV = cv2.cvtColor( imageRGB, cv2.COLOR_BGR2Luv )

imageLUV2RGB = cv2.cvtColor( imageLUV, cv2.COLOR_Luv2BGR )

meanShift = meanShiftSeg( imageLUV, 1 )

segImage = meanShift.applyMeanShift

tShit = meanShift.getSegmentedImage
print  type(tShit)

# cv2.imshow( 'image', imageLUV2RGB )
# print type( imageLUV ), np.shape( imageLUV ), imageLUV.max(), imageLUV.min()
#
# featureVec = imageLUV.reshape( -1 ,3 )
# # print np.shape(featureVec)
# luvHist , edges = np.histogramdd( featureVec )
# print np.max(imageLUV[:,:,2])
# print type(luvHist), luvHist.shape, type(edges), np.shape(edges), luvHist
# plt.xlim([ 0,256 ])
# plt.plot(luvHist)
# plt.show()
# cv2.imshow('image', luvHist)

# window = np.array([[0,2,1], [4,0,5], [ 11,5,9 ]])
# idx = range( 3 )
# nonZeroIdx = np.transpose(np.nonzero( window ))
# # print nonZeroIdx
# totalMass = np.max(np.cumsum( window ))
# print totalMass
# momentCol = np.max(np.cumsum(window.cumsum( axis=1 )[:,3-1]*idx))
# print momentCol
# cntrCol = np.round(1.0*momentCol/totalMass)
# print 1.0*momentCol/totalMass,cntrCol


# print 1.0*256/25, 256/32
# print a
# print np.transpose(np.nonzero(a))
# print np.transpose(np.nonzero(a))[:,0]
# print a.cumsum( axis=0 )[a.shape[0]-1]
# print a.cumsum( axis=0 )[a.shape[0]-1]*np.transpose(np.nonzero(a))[:,0]

# a = np.array([[0,2,1], [4,0,5], [ 11,5,9 ]])
# idx = range( 3 )
# print a
# # print np.transpose(np.nonzero(a))
# # print np.transpose(np.nonzero(a))[:,0]
# print a.cumsum( axis=1 )[:,2]
# print a.cumsum( axis=1 )[:,2]*idx
#

cv2.waitKey(0)
cv2.destroyAllWindows()



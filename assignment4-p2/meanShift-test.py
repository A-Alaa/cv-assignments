from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile , join
import numpy as np
import cv2
import sys
sys.path.append('../scripts')

from hisMeanShift import meanShiftSeg

imagePath = "../images/assignment4/seg1.jpg"

imageRGB = cv2.imread( imagePath )

imageLUV = cv2.cvtColor( imageRGB, cv2.COLOR_RGB2LUV )

imageLUV2RGB = cv2.cvtColor( imageLUV, cv2.COLOR_LUV2RGB )
# print type(imageLUV)

meanShift = meanShiftSeg( imageRGB, 7 )
segImage = meanShift.applyMeanShift()

# # print type(segImage)
# # print np.shape(segImage)
# segImageRGB = cv2.cvtColor( segImage, cv2.COLOR_LUV2RGB )
# segimTem = cv2.cvtColor( segImageRGB, cv2.COLOR_LUV2BGR )
cv2.imshow( 'image', segImage )
# plt.subplot(211)
# plt.imshow( imageRGB )
# plt.subplot(212)
# plt.imshow( imageRGB )
# plt.show()

# zero = np.zeros((4,4))
# print np.max(zero.cumsum())


cv2.waitKey(0)
cv2.destroyAllWindows()



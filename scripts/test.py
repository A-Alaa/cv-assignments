import numpy as np
import sys
import cv2

imageColor = cv2.imread('../images/assignment3/Pyramids1.jpg')
imageGray = cv2.imread('../images/assignment3/Pyramids1.jpg', 0)
sys.path.append('../scripts')

from mySIFT import mySIFT

x = np.array(  [[1,2,3] , [4,5,6] , [7,8,9]] )
i=0
y = x[(i+1):3,(i+1):3]
print y
print y-1

z=[1 ,2 ,3]
print z[-1]

corners = [(20,20)]

m = mySIFT( imageGray , corners )
m.getSIFTDescriptors()
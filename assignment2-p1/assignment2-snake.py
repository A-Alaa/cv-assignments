from os import listdir
from os.path import isfile , join
import numpy as np
import cv2
import sys

sys.path.append('../scripts')

image = np.matrix([ np.arange(4) ,
                    np.arange(4 , 8) ,
                    np.arange(8 , 12) ,
                    np.arange(12 , 16) ] , dtype = np.float32)
print image

points = [ (2 , 1) , (3 , 1) , (3 , 0) , (2 , 0) ]

pts = np.asarray( points , dtype = np.float32 )
print pts[0]
print pts[-1]
delta = pts[ 0 , :] - pts[ -1 , :]
d = ( delta[0]**2 + delta[1]**2)**(0.5)
print d

points = np.asarray(points)
distance = np.sqrt(np.square(points[ : , 0 ] - np.roll(points[ : , 0 ] , -1)) +
                   np.square(points[ : , 1 ] - np.roll(points[ : , 1 ] , -1)))
print distance
print "Image points"
imagePoints = [ image[ r , c ] for r , c in points ]
imagePoints = (imagePoints - min(imagePoints)) / (
max(imagePoints) - min(imagePoints))
print imagePoints

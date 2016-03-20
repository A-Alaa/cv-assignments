# import the necessary packages
from os import listdir
from os.path import isfile , join
import numpy as np
import cv2
import sys

sys.path.append('../scripts')

import myCanny
import myHoughCircles

# Get file names in "./images" directory
imageFiles = [join("../images/assignment2" , f) for f in
              listdir("../images/assignment2") if
              isfile(join("../images/assignment2" , f))]
imageFile = "../images/assignment2/Apple2.jpg"

# Load the images
# Gray image is calculated as :
# Y = 0.299 R + 0.587 G + 0.114 B
image = cv2.imread(imageFile)
image_GRAY = cv2.imread(imageFile , cv2.IMREAD_GRAYSCALE)


# Convert from uint8 precision to double precision.
image_GRAY = np.float64(image_GRAY)

# Normalize grayscale images from 0.0 to 1.0.
image_GRAY = cv2.normalize(image_GRAY , image_GRAY , 0 , 1 , cv2.NORM_MINMAX ,
                           cv2.CV_64F)

cannyImage = myCanny.myCanny(image_GRAY , 3 , 1.5 , 0.4 , 0.1)

circles = myHoughCircles.myHoughCircles(cannyImage ,150)

for x,y,r in circles:
    # draw the outer circle
    cv2.circle(image,(x,y),r,(0,255,0),1)
    # draw the center of the circle
    cv2.circle(image,(x,y),2,(0,0,255),2)


# Edge Image
cv2.namedWindow("Canny Image" , cv2.WINDOW_NORMAL)
cv2.imshow("Canny Image" , cannyImage)
cv2.waitKey()

# Image + Circle
cv2.namedWindow("Canny Image" , cv2.WINDOW_NORMAL)
cv2.imshow("Canny Image" , image )
cv2.waitKey()

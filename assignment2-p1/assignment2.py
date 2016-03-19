# import the necessary packages
from os import listdir
from os.path import isfile , join
import numpy as np
import cv2
import sys

sys.path.append('../scripts')

import myCanny
import myHoughLine

# Get file names in "./images" directory
imageFiles = [join("../images/assignment2" , f) for f in
              listdir("../images/assignment2") if
              isfile(join("../images/assignment2" , f))]
imageFile = "../images/assignment2/Lines.jpg"

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

cannyImage = myCanny.myCanny(image_GRAY , 3 , 1.0 , 0.2 , 0.2)

lines = myHoughLine.myHoughLines(cannyImage , np.pi / 180 , 200)
#
# lines = cv2.HoughLines(np.uint8(np.abs(cannyImage) * 255) , 1 ,
#                              1 / np.pi , 50)
linesOnlyDraft = np.zeros(image.shape , dtype = np.uint8)

if lines is None :
    print "No lines detected"
    exit(1)
for theta , rho in lines :
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv2.line(linesOnlyDraft , (x1 , y1) , (x2 , y2) , (0 , 255 , 0) , 2)




linesOnlyImage = np.zeros(image.shape , dtype = np.uint8)

cannyImg_RGB = cv2.cvtColor(np.uint8(abs(cannyImage) * 255) ,
                            cv2.COLOR_GRAY2BGR)

cv2.bitwise_and(linesOnlyDraft , cannyImg_RGB ,
                linesOnlyImage)

image = cv2.addWeighted(image , 0.5 , linesOnlyImage , 0.5 , 0)

# Edge Image
cv2.namedWindow("Canny Image" , cv2.WINDOW_NORMAL)
cv2.imshow("Canny Image" , cannyImage)
cv2.waitKey()

# Lines Only Image
cv2.namedWindow("Line Only" , cv2.WINDOW_NORMAL)
cv2.imshow("Line Only", linesOnlyDraft)
cv2.waitKey()

# (Lines Only Image) AND (Edge Image)
cv2.namedWindow("Lines Only AND Edge Image" , cv2.WINDOW_NORMAL)
cv2.imshow("Lines Only AND Edge Image" , linesOnlyImage)
cv2.waitKey()

# (Lines Only Image) AND (Edge Image) + Original Image
cv2.namedWindow("Lines Only AND Edge Image + Image" , cv2.WINDOW_NORMAL)
cv2.imshow("Lines Only AND Edge Image + Image" , image)
cv2.waitKey()

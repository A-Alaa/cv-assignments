# import the necessary packages
from os import listdir
from os.path import isfile , join
import numpy as np
import cv2
import sys

sys.path.append('../scripts')

import myCV

# Get file names in "./images" directory
imageFiles = [join("../images/assignment2" , f) for f in
              listdir("../images/assignment2") if
              isfile(join("../images/assignment2" , f))]

# Load the images
# Gray image is calculated as :
# Y = 0.299 R + 0.587 G + 0.114 B
images = [cv2.imread(imageFile , cv2.IMREAD_GRAYSCALE) for imageFile in
          imageFiles]

# Convert from uint8 precision to double precision.
images = [np.float64(image) for image in images]

# Normalize grayscale images from 0.0 to 1.0.
images = [cv2.normalize(img , img , 0 , 1 , cv2.NORM_MINMAX , cv2.CV_64F) for
          img in images]

cannyImages = [myCV.myCanny(img , 3 , 1.0 , 0.2 , 0.8) for img in images]

for img , imgFile in zip(cannyImages , imageFiles) :
    cv2.namedWindow(imgFile , cv2.WINDOW_NORMAL)
    cv2.imshow(imgFile , np.uint8(np.abs(img) * 255))

cv2.waitKey()
cv2.destroyAllWindows()

from os import listdir
from os.path import isfile , join
import numpy as np
import cv2
import sys

sys.path.append('../scripts')

from KmeansSegmentation import KmeansSegmentation

# Get file names in "./images" directory
imageFiles = [join("../images/assignment4" , f) for f in
              listdir("../images/assignment4") if
              isfile(join("../images/assignment4" , f))]

image = cv2.imread(imageFiles[3])
image = cv2.cvtColor( image , cv2.COLOR_RGB2Luv  )

KmeansSegmentation( image , 2  )

featureVector = image.reshape((-1 , 3))
featureVector = np.float32(featureVector)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER , 10 , 1.0)
ncluster = 2
compactness , labels , centers = cv2.kmeans(featureVector , ncluster , None ,
                                            criteria ,
                                            2 , cv2.KMEANS_RANDOM_CENTERS)
centers = np.uint8( centers )
output  = centers[ labels.flatten() ]
output  = output.reshape(( image.shape ))
# print centers
# print featureVector[:5]
#
# print image.shape
# print featureVector.shape

cv2.imshow('image' , image)
cv2.imshow('output' , output )
cv2.waitKey(0)
cv2.destroyAllWindows()

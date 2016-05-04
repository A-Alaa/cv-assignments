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

image = cv2.imread(imageFiles[12])
# image = cv2.cvtColor( image , cv2.COLOR_RGB2Luv  )

labels , centroids , wss = KmeansSegmentation( image , 4  )

centers = np.uint8( centroids )
output  = centers[ labels.flatten() ]
output  = output.reshape(( image.shape ))

cv2.imshow('image' , image)
cv2.imshow('output' , output )
cv2.waitKey(0)
cv2.destroyAllWindows()

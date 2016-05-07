from os import listdir
from os.path import isfile , join
import cv2
import sys
import time as time
import numpy as np

sys.path.append('../scripts')
from AgglomerativeClustering import AgglomerativeClustering


# Get file names in "./images" directory
imageFiles = [join("../images/assignment4" , f) for f in
              listdir("../images/assignment4") if
              isfile(join("../images/assignment4" , f))]

image = cv2.imread("../images/assignment4/seg1.jpg" , 0 )
# image = cv2.cvtColor( image , cv2.COLOR_RGB2Luv  )
print image.shape
# AgglomerativeClustering( image )

featureVector = image.reshape(-1,1)

print("Compute unstructured hierarchical clustering...")
st = time.time()
x = AgglomerativeClustering( image )
elapsed_time = time.time() - st
print("Elapsed time: %.2fs" % elapsed_time)


cv2.imshow('image' , image)
cv2.imshow('output' , np.uint8( x.image) )
cv2.waitKey(0)
cv2.destroyAllWindows()

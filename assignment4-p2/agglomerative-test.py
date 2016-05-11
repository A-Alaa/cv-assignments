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

image = cv2.imread("../images/assignment4/seg1.jpg" )
image = cv2.cvtColor( image , cv2.COLOR_RGB2Luv  )

imageCopy =  np.array( image , copy = True )
featureVector = np.array( imageCopy[ : , : , 2 ] , copy = True )
featureVector = featureVector.reshape(-1,1)

print("Compute unstructured hierarchical clustering...")
st = time.time()
agglomerative = AgglomerativeClustering( featureVector , nclusters = 7 )

labels = agglomerative.labels
classes,counts = np.unique( labels , return_counts =  True )
labels = labels.reshape( image.shape[0] , image.shape[1] )
image = cv2.cvtColor( image , cv2.COLOR_Luv2RGB  )
imageCopy = cv2.cvtColor( imageCopy , cv2.COLOR_Luv2RGB  )

print labels.shape
print classes , counts

for i in range(classes.shape[0]) :
    imageCopy[ labels == classes[i] , : ] = \
        np.mean(  imageCopy[ labels == classes[i] , : ]  )

elapsed_time = time.time() - st
print("Elapsed time: %.2fs" % elapsed_time)




cv2.imshow('image' , image)
cv2.imshow('output' ,  imageCopy )
cv2.waitKey(0)
cv2.destroyAllWindows()

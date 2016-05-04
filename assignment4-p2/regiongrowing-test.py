from os import listdir
from os.path import isfile , join
import numpy as np
import cv2
import sys

sys.path.append('../scripts')
from RegionGrowingSegmentation import RegionGrowingSegmentation

# Get file names in "./images" directory
imageFiles = [ join("../images/assignment4" , f) for f in
               listdir("../images/assignment4") if
               isfile(join("../images/assignment4" , f)) ]

image = cv2.imread(imageFiles[12] , 0)
coloredImage = cv2.imread( imageFiles[12] )

print image.shape , image.dtype
print coloredImage.shape , coloredImage.dtype


segmentedImage = RegionGrowingSegmentation(image)


def getxy( event , x , y , flags , param ) :
    if event == cv2.EVENT_LBUTTONDOWN :
        seedPoint = (y , x)
        label = segmentedImage.newRegion(seedPoint)
        coloredImage[ segmentedImage.labelImage == label ] = \
            np.array( np.random.choice(255 ,3 ))

        cv2.imshow('image' , coloredImage)


# Set mouse CallBack event
cv2.namedWindow('image' , cv2.WINDOW_NORMAL)
cv2.setMouseCallback('image' , getxy)
cv2.imshow('image' , coloredImage )

cv2.waitKey(0)
cv2.destroyAllWindows()

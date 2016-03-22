import cv2
import numpy as np
import copy as cp
from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join
import numpy as np
import sys

sys.path.append('../scripts')

from Snake import GreedySnake

# ____________________________________________________________________________________________________________________#
cv2.__version__
# ____________________________________________________________________________________________________________________#
global imageColor, imageGray, imageContour
imageColor = cv2.imread('../images/assignment2/shell1.jpg')
imageGray = cv2.imread('../images/assignment2/shell1.jpg', 0)
imageContour = cp.deepcopy(imageColor)
contourPoints = np.zeros(1)
contourList = []


def getxy(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        contourList.append((y, x))
        # cv2.drawContours( imageContour, contour, -1, ( 128,128,128 )  )
        print "(row, col) = ", (y, x)
        cv2.drawMarker(imageColor, (x, y), (0, 0, 0))
        cv2.imshow('image', imageColor)
    else:
        if event == cv2.EVENT_RBUTTONDOWN:
            contourPoints = np.array(contourList)

            startSnake( contourPoints )
            # contourPoints[1][1] = 5
            # print contourPoints
            # snakeInternal(1,1  )



def startSnake( initialContour ):
    # alpha = 1, beta=1, gamma=1
    snake = GreedySnake(2.5, 2.5, 0.05  , initialContour, imageGray)

    cv2.namedWindow('snake-image',cv2.WINDOW_NORMAL)
    while( snake.iterate() >= 0 ) :
        newContour = snake.getContour()
        newContour = [ (c,r) for (r,c) in newContour ]
        newContour = np.array( newContour )
        imageSnake = np.array( imageColor , copy = True )
        imageSnake = cv2.drawContours( imageSnake,[newContour],-1, ( 255,0,0 ) , 5)

        cv2.imshow('snake-image',imageSnake)
        cv2.waitKey(0)

    exit(1)
# Set mouse CallBack event
cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.setMouseCallback('image', getxy)
cv2.imshow('image', imageColor)

cv2.waitKey(0)
cv2.destroyAllWindows()

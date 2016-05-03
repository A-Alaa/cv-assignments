import cv2
import numpy as np
import copy as cp
from matplotlib import pyplot as plt
#____________________________________________________________________________________________________________________#
cv2.__version__
#____________________________________________________________________________________________________________________#
global imageColor, imageGray, imageContour
imageColor = cv2.imread('Regular-Shapes.jpg')
imageGray = cv2.imread( 'Regular-Shapes.jpg', 0 )
imageContour = cp.deepcopy( imageColor )
contourPoints = np.zeros(1)
contourList = []
# contourPoints = np.array([])

def circShift( point, dir, size ):
    shiftedPoint = (point + dir + size )%size
    return shiftedPoint


def energyCont( contour, x, y, alpha ):
    avgDist = 0
    pointsNumber = np.shape( contour )[0]
    # kernelIndxs = [ [[ y-1, x-1 ],[ y-1, x ],[ y-1, x+1 ]]
    #            [[ y, x-1 ],[ y, x ],[ y, x+1 ]]
    #            [[ y+1, x-1 ],[ y+1, x ],[ y+1, x+1 ]]]
    kernel = np.array( [[ 1, 1, 1 ],
                        [ 1, 1, 1 ],
                        [ 1, 1, 1 ]], dtype=float)
    for point in range( pointsNumber ):
        currentPointX = contour[point][0]
        prevPointX = circShift( currentPointX, -1, pointsNumber )
        currentPointY = contour[point][1]
        prevPointY = circShift( currentPointY, -1, pointsNumber )
        avgDist += np.sqrt( ( currentPointX - prevPointX )*( currentPointX - prevPointX ) +
                            ( currentPointY - prevPointY )*( currentPointY - prevPointY ))
        avgDist /= pointsNumber

    for kernelY in range( 3 ):
        for kernelX in range( 3 ):
            xPrev = circShift( x, -1, pointsNumber )
            yPrev = circShift( y, -1, pointsNumber )
            dist = np.sqrt( ( (kernelX+x-1) - xPrev )*( (kernelX+x-1) - xPrev )+
                            ( (kernelY+y-1) - yPrev )*( (kernelY+y-1) - yPrev ) )
            kernel[ kernelY, kernelX ] = np.abs( avgDist - dist )*alpha
        kernel/kernel.max()
    return kernel

def energyCurv( contour, x, y, beta ):
    pointsNumber = np.shape( contour )[0]
    # kernelIndxs = [ [[ y-1, x-1 ],[ y-1, x ],[ y-1, x+1 ]]
    #            [[ y, x-1 ],[ y, x ],[ y, x+1 ]]
    #            [[ y+1, x-1 ],[ y+1, x ],[ y+1, x+1 ]]]
    kernel = np.array( [[ 1, 1, 1 ],
                        [ 1, 1, 1 ],
                        [ 1, 1, 1 ]], dtype=float)
    for kernelY in range( 3 ):
        for kernelX in range( 3 ):
            xPost = circShift( x, 1, pointsNumber )
            yPost = circShift( y, 1, pointsNumber )
            xPrev = circShift( x, -1, pointsNumber )
            yPrev = circShift( y, -1, pointsNumber )
            curvComp = np.sqrt( (  xPost - 2*(kernelX+x-1) + xPrev )*( xPost - 2*(kernelX+x-1) + xPrev )+
                            ( yPost - 2*(kernelY+y-1) + yPrev )*( yPost - 2*(kernelY+y-1) + yPrev ) )
            kernel[ kernelY, kernelX ] = curvComp*beta
        kernel/kernel.max()
    return kernel


def snakeInternal( alpha, beta ):
    # kernelFinal = np.array( [[ 1, 1, 1 ],
    #                          [ 1, 1, 1 ],
    #                          [ 1, 1, 1 ]], dtype=float)
    for contourItem in range( contourPoints.shape[0]):
        x = contourPoints[contourItem][1]
        y = contourPoints[contourItem][0]
        kernelFinal = energyCont( contourPoints,x,y,alpha ) + energyCurv( contourPoints,x,y,beta )
        finalX = x + np.unravel_index( kernelFinal.argmax(), kernelFinal.shape )[1] -1
        finalY = y + np.unravel_index( kernelFinal.argmax(), kernelFinal.shape )[0] -1
        contourPoints[ contourItem][0 ] = finalY
        contourPoints[ contourItem][1 ] = finalX

    # return contour




def getxy(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN :
        contourList.append( (y,x) )
        # cv2.drawContours( imageContour, contour, -1, ( 128,128,128 )  )
        print "(row, col) = ", (y,x)
        cv2.drawMarker( imageColor, ( x,y ), ( 0,255,0 )  )
        cv2.imshow( 'image', imageColor )
    else:
        if event == cv2.EVENT_RBUTTONDOWN :
            contourPoints = np.array( contourList )
            print contourPoints.shape
            print contourPoints
            # contourPoints[1][1] = 5
            # print contourPoints
            # snakeInternal(1,1  )

def startSnake( contour  ) :




#Set mouse CallBack event
cv2.namedWindow('image')
cv2.setMouseCallback('image', getxy)
cv2.imshow('image', imageColor)

# a = np.array( shape=(2,3) )

# array = np.array([[5, 5, 2],
#                   [0,1 , 3],
#                   [4, 2, 3]], dtype=float )
#
#
# print np.unravel_index( array.argmin(),  array.shape )
# print array/array.max()


cv2.waitKey(0)
cv2.destroyAllWindows()

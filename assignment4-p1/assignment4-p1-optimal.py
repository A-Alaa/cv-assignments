import cv2 as cv
import sys
sys.path.append('../scripts')
from OptimalThresholding import optimal_thresholding

# load gray images
imageColor = cv.imread('../images/assignment4/0019hv3.bmp')
imageGray = cv.imread('../images/assignment4/0019hv3.bmp', cv.IMREAD_GRAYSCALE)

# threshold gray image using optiaml thresholding method
thresholded_image , optimal_threshold = optimal_thresholding(imageGray)

print "optimal threshold = ", optimal_threshold

cv.namedWindow('Original image',cv.WINDOW_NORMAL)
cv.imshow('Original image', imageGray)

cv.imshow('Thresholded image using optimal thresholding', thresholded_image)

cv.waitKey(0)
cv.destroyAllWindows()


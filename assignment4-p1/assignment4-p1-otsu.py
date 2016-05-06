import cv2 as cv
import sys
sys.path.append('../scripts')
from OtsuThresholding import otsu_thresholding

# load gray images
imageColor = cv.imread('../images/assignment4/0019hv3.bmp')
imageGray = cv.imread('../images/assignment4/0019hv3.bmp', cv.IMREAD_GRAYSCALE)

# threshold gray image using optiaml thresholding method
thresholded_image , otsu_threshold = otsu_thresholding(imageGray)

print "otsu threshold = ", otsu_threshold

cv.namedWindow('Original image',cv.WINDOW_NORMAL)
cv.imshow('Original image', imageGray)

cv.imshow('Thresholded image using Otsu thresholding', thresholded_image)

cv.waitKey(0)
cv.destroyAllWindows()


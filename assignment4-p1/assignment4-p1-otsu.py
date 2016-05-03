import cv2 as cv
import sys
sys.path.append('../scripts')
from OtsuThresholding import otsuThresholding

# load gray images
imageColor = cv.imread('../images/assignment4/MRIbrain2.jpg')
imageGray = cv.imread('../images/assignment4/MRIbrain2.jpg', cv.IMREAD_GRAYSCALE)

# threshold gray image using optiaml thresholding method
thresholded_image , otsu_threshold = otsuThresholding(imageGray)

print "otsu threshold", otsu_threshold

cv.namedWindow('Original image',cv.WINDOW_NORMAL)
cv.imshow('Original image', imageGray)

cv.imshow('Thresholded image using Otsu thresholding', thresholded_image)

cv.waitKey(0)
cv.destroyAllWindows()


# import the necessary packages
from os import listdir
from os.path import isfile , join
import numpy as np
from scipy.signal import convolve2d
import cv2
import copy

# Get file names in "./images" directory
imageFiles = [ join("../images/assignment1" , f) for f in
               listdir("../images/assignment1") if
               isfile(join("../images/assignment1" , f)) ]

# Load the images
# Gray image is calculated as :
# Y = 0.299 R + 0.587 G + 0.114 B
images = [ cv2.imread(imageFile , cv2.IMREAD_GRAYSCALE) for imageFile in
           imageFiles ]

# Convert from uint8 precision to double precision.
images = [ np.float64(image) for image in images ]

normalizedImages = copy.deepcopy(images)
for image , nimage in zip(images , normalizedImages) :
    cv2.normalize(image , nimage , 0 , 1 , cv2.NORM_MINMAX , cv2.CV_64F)
images = normalizedImages
# ---------------------------------------------------------------- #
# 1.Sobel: Apply filter in both horizontal and vertical directions.
sobelHKernel = np.array([ [ -1 , 0 , 1 ] ,
                          [ -2 , 0 , 2 ] ,
                          [ -1 , 0 , 1 ] ])

sobelVKernel = sobelHKernel.transpose()

images_sobelH = [ convolve2d(image , sobelHKernel) for image in
                  images ]

images_sobelV = [ convolve2d(image , sobelVKernel) for image in
                  images ]

# ---------------------------------------------------------------- #
# 2.Roberts cross edge detection
robertKernel_315 = np.array([ [ 1 , 0 ] ,
                              [ 0 , -1 ] ])

robertKernel_45 = np.array([ [ 0 , -1 ] ,
                             [ 1 , 0 ] ])

images_roberts = [ convolve2d(image , robertKernel_45) for image in
                   images ]
images_roberts = [ convolve2d(image , robertKernel_315) for image in
                   images_roberts ]

# ---------------------------------------------------------------- #
# 3.Prewitt edge detection
prewittKernelH = np.array([ [ -1 , 0 , 1 ] ,
                            [ -1 , 0 , 1 ] ,
                            [ -1 , 0 , 1 ] ])
prewittKernelV = prewittKernelH.transpose()

imagesPrewittH = [ convolve2d(image , prewittKernelH) for image in
                   images ]
imagesPrewittV = [ convolve2d(image , prewittKernelV) for image in
                   images ]

# ---------------------------------------------------------------- #
# 4.Canny Edge Detection:
# 4.1 Smoothing
gaussianKernel1D = cv2.getGaussianKernel(3 , 1.0)
gaussianKernel2D = gaussianKernel1D.transpose() * gaussianKernel1D

smoothedImages = [ convolve2d(img , gaussianKernel2D) for img in images ]

# 4.2 Gradients
gradientImages = [ ]
phaseImages = [ ]

for image in smoothedImages :
    sobelH = convolve2d(image , sobelHKernel)
    sobelV = convolve2d(image , sobelVKernel)
    phase = np.arctan2(sobelH , sobelV) * (180.0 / np.pi)

    # Assign phase values to nearest [ 0 , 45 , 90 ,  135 ]
    phase = ((45 * np.round(phase / 45.0)) + 180) % 180;

    gradientImages.append(np.sqrt(sobelH * sobelH + sobelV * sobelV))
    phaseImages.append(phase)

# 4.3 Non-Maximum Supresssion
thinEdgedImages = copy.deepcopy(gradientImages)
for gradient , phase in zip(thinEdgedImages , phaseImages) :
    for x in range(1 , gradient.shape[ 0 ] - 1) :
        for y in range(1 , gradient.shape[ 1 ] - 1) :

            # If already supressed, proceed to the next pixel.
            if gradient[ x , y ] == 0 :
                continue

            theta = phase[ x , y ]
            maxGradient = gradient[ x , y ];

            if theta == 0 :
                maxGradient = max([ gradient[ x - 1 , y ] ,
                                    gradient[ x , y ] ,
                                    gradient[ x + 1 , y ] ])

            elif theta == 45 :
                maxGradient = max([ gradient[ x - 1 , y - 1 ] ,
                                    gradient[ x , y ] ,
                                    gradient[ x + 1 , y + 1 ] ])


            elif theta == 90 :
                maxGradient = max([ gradient[ x , y - 1 ] ,
                                    gradient[ x , y ] ,
                                    gradient[ x , y + 1 ] ])


            elif theta == 135 :
                maxGradient = max([ gradient[ x + 1 , y - 1 ] ,
                                    gradient[ x , y ] ,
                                    gradient[ x - 1 , y + 1 ] ])
            else :
                print("Unexpected theta value")
                print theta
                exit(1)

            if gradient[ x , y ] < maxGradient :
                gradient[ x , y ] = 0

# 4.4 Double Thresholding
doubleThresholdImages = copy.deepcopy(thinEdgedImages)
maxThreashold = 0.8
minThreshold = 0.2

for image in doubleThresholdImages :
    for x in range(0 , image.shape[ 0 ]) :
        for y in range(0 , image.shape[ 1 ]) :
            if image[ x , y ] < minThreshold :
                image[ x , y ] = 0.0
            elif image[ x , y ] >= maxThreashold :
                image[ x , y ] = 1.0

# 4.5 Edge Trackign by hysteresis
edgeTrackingImages = copy.deepcopy(doubleThresholdImages)
for image in edgeTrackingImages :
    for x in range(1 , image.shape[ 0 ] - 1) :
        for y in range(1 , image.shape[ 1 ] - 1) :

            if image[ x , y ] > maxThreashold :
                image[ x , y ] = 1.0

            elif image[ x , y ] < minThreshold :
                image[ x , y ] = 0

            else :
                maxGradient = max([ [ image[ x - 1 , y - 1 ] ,
                                      image[ x - 1 , y ] ,
                                      image[ x - 1 , y + 1 ] ,
                                      image[ x , y - 1 ] ,
                                      image[ x , y + 1 ] ,
                                      image[ x + 1 , y - 1 ] ,
                                      image[ x + 1 , y ] ,
                                      image[ x + 1 , y + 1 ] ] ])
                if maxGradient >= maxThreashold :
                    image[ x , y ] = maxThreashold
                else :
                    image[ x , y ] = 0

cannyImages = [
    cv2.Canny(np.uint8(np.abs(img) * 255) , 50 , 200 , L2gradient = 1) for
    img in
    images ]
for img , imgFile in zip(edgeTrackingImages , imageFiles) :
    cv2.namedWindow(imgFile , cv2.WINDOW_NORMAL)
    cv2.imshow(imgFile , np.uint8(np.abs(img) * 255))

cv2.waitKey()
cv2.destroyAllWindows()

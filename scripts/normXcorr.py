import numpy as np
from scipy.signal import convolve
import cv2


# Modified version of NormXCorrelation to work on 128-SIFT descriptor
# This version tries to find the 128-feature between the array of 128-features
# The template is the feature need to correlate it is  1 X 128 vector  (rows X columns)
# the image is the array of the features it is  n-features X 128 (rows X columns)
class NormXCorrMod :
    def __init__(self, template, image, keyPointIndex):
        self.image = np.array( image )
        self.template = template
        self.templateFlat = np.array( template ).flatten()
        self.imageFlat = np.array( image ).flatten()
        self.KPIndex = np.array(keyPointIndex)
        self.corrIndex = 0


    def __normXCorr__(self):
        template0Mean = self.templateFlat - self.templateFlat.mean()
        varTemplate = self.templateFlat.var()
        # imageMean =self.imageFlat.mean()
        image0Mean = self.image[:,] - self.image.mean( 0 )
        varImage = self.image.var( 0 )
        imageTempCoVar = varImage*varTemplate
        corrValue = []
        # Perfoem the normXcorelation
        for featIdx in range(0, self.image.shape[0]):
            templateXimage = np.sum(np.multiply( template0Mean, image0Mean[featIdx]), axis=0 )
            templateXimageNorm = templateXimage/imageTempCoVar[featIdx]
            corrValue.append(templateXimageNorm)

        corrValue = np.array( corrValue )
        self.corrIndex = self.KPIndex[corrValue.argmax()]

    def getCorrIndex(self):
        self.__normXCorr__()
        return self.corrIndex





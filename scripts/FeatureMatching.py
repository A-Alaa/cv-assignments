import cv2
import numpy as np

class FeatureMatcher:
    def __init__(self, features1,corners1, features2,corners2):
        self.features1 =features1
        self.features2 =features2
        self.corners1 = corners1
        self.corners2 = corners2
        self.matchingCorners2 = []
        self.result = []
        self.__SSD__()


    def __SSD__(self) :
        result = []
        for feature1 in self.features1 :
            for feature2 in self.features2 :
                ssd = np.sum((feature1-feature2)**2)
                result.append(ssd)

        # reshape SSD list  into a matrix
        # Compare 10 features from each image
        ssdMatrix = np.reshape(result, (10,10))

        # Get minimum SSD along each row
        minIndices = ssdMatrix.argmin(axis=1)

        # Add each matching feature at corresponding min index to   result
        for index in minIndices :
            self.matchingCorners2.append(self.corners2[index])
        self.result = [self.corners1 , self.matchingCorners2]

    def getMatchingPoints(self):
        return self.result

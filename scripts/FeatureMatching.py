import cv2
import numpy as np

class FeatureMatcher:
    def __init__(self, features1,corners1, features2,corners2):
        self.features1 = features1
        self.features2=features2
        self.corners1 = corners1
        self.corners2 = corners2
        self.matchingCorners2 = []
        self.result = []
        self.__SSD__()

    def __SSD__(self) :
        # print "f1",len(self.features1)
        #
        # print "f2", len(self.features2)
        #
        # print "f3", self.corners1
        #
        # print "f4", self.corners2
        result = []
        for feature1 in self.features1 :
            for feature2 in self.features2 :
                ssd = np.sum((feature1-feature2)**2)
                result.append(ssd)

        # reshape SSD into a matrix
        ssdMatrix = np.reshape(result, (-1, 10))

        # get minimum SSD along each row
        minIndices = ssdMatrix.argmin(axis=1)
        #print "min" , minIndices

        # add each matching feature at corresponding min index to   result

        for index in minIndices :
            self.matchingCorners2.append(self.corners2[index])

        self.matchingCorners2 =np.reshape(self.matchingCorners2,(-1,2))

        # print self.matchingCorners1
        # print len(self.matchingCorners1)
        #
        # print self.matchingCorners2
        # print len(self.matchingCorners2)

        self.result = [self.corners1 , self.matchingCorners2]
        # print self.result

    def getMatchingPoints(self):
        return self.result




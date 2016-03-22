import numpy as np
from scipy.signal import convolve2d
import cv2

class GreedySnake:
    def __init__(self, alpha, beta, gamma, initialContour, image):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.image = np.matrix(image, copy=True, dtype=np.float32)
        self.gradientImage = self.__getEnergyImage__()
        self.contour = np.asarray(initialContour)
        self.movements = 0

    def iterate(self):
        """
        Iterates over the contour points to minimize the cost.
        :return: count of movements occurred.
        """
        lastMovementCount = self.movements

        for contourIdx in range(self.contour.shape[0]):
            self.__minimizeCostAt__(contourIdx)

        return self.movements - lastMovementCount

    def getContour(self):
        return self.contour

    def show(self):
        pass

    def __getElasticityAt__(self, contourIndex):
        # E(continuity) = || mean(distance) - |P(i) - P(i-1)| ||^2
        """

        :param contourIndex:
        :return:
        """
        dy = self.contour[:, 0] - np.roll(self.contour[:, 0], -1)
        dx = self.contour[:, 1] - np.roll(self.contour[:, 1], -1)
        distance = np.sqrt(np.square(dx) + np.square(dy))
        return np.square(np.mean(distance) - distance[contourIndex])

    def __getCurvatureAt__(self, contourIdx):
        # E(curvature) = || P(i-1) - 2 P(i) + P(i+1) ||^2
        """

        :param contourIdx:
        :return:
        """
        maxIdx = self.contour.shape[0]

        curvature = self.contour[contourIdx - 1, :] - \
                    2 * self.contour[contourIdx, :] + \
                    self.contour[(contourIdx + 1)%maxIdx, :]

        return curvature[0] ** (2) + curvature[1] ** (2)

    def __getExternalEnergyAt__(self, neighbourWindow):
        """

        :param neighbourWindow:
        :return:
        """
        gradient = [self.gradientImage[r, c] for r, c in neighbourWindow]

        # Normalize Contour Gradient.
        gradient = (gradient - max(gradient)) / (max(gradient) - min(gradient))
        return gradient

    def __getEnergyImage__(self):
        # E(image) = - || Gaussian * Gradient( Intensity ) ||
        # Apply Gaussian Blurring.
        # Hard-coded Gaussian Kernel. order=3, sigma=1.
        gaussianKernel1D = np.matrix([0.27406862, 0.45186276, 0.27406862])
        gaussianKernel2D = gaussianKernel1D.transpose() * gaussianKernel1D
        gaussianImage = convolve2d(self.image, gaussianKernel2D,
                                   mode='same')

        gaussianImage = cv2.GaussianBlur( self.image , (15,15) , sigmaX= 5 , sigmaY= 5)
        # Apply Sobel Gradient in both V/H directions.
        sobelHKernel = np.array([[-1, 0, 1],
                                 [-2, 0, 2],
                                 [-1, 0, 1]])
        sobelH = convolve2d(gaussianImage, sobelHKernel, mode='same')
        sobelV = convolve2d(gaussianImage, sobelHKernel.transpose(),
                            mode='same')


        return - np.sqrt(np.square(sobelH) + np.square(sobelV))

    def __minimizeCostAt__(self, contourIndex):
        # Get the current (r)ow and (c)olumn.
        r = self.contour[contourIndex, 0]
        c = self.contour[contourIndex, 1]

        # Get all neighbors.
        neighbours = [(r - 1, c - 1), (r - 1, c), (r - 1, c + 1),
                      (r, c - 1), (r, c), (r, c + 1),
                      (r + 1, c - 1), (r + 1, c), (r + 1, c + 1)]

        # Calculate cost at each neighbour.
        cost = self.__getCost__(contourIndex, neighbours)

        # Minimize cost.
        minCostIdx = np.argmin( cost )

        # Update the contour to the new neighbor.
        newPoint = neighbours[ minCostIdx ]
        # if newPoint[0] > self.image.shape[0]-3 or newPoint[0] < 2  or\
        #     newPoint[1] > self.image.shape[1]-3 or newPoint[1] < 2 :
        #     self.contour[contourIndex, 0] = r
        #     self.contour[contourIndex, 1] = c

        if newPoint != (r, c):
            self.contour[contourIndex, 0] = newPoint[0]
            self.contour[contourIndex, 1] = newPoint[1]
            self.movements += 1
        else:
            # Else, return the contour to its old state.
            self.contour[contourIndex, 0] = r
            self.contour[contourIndex, 1] = c

    def __getCost__(self, contourIndex, neighbours):
        """

        :param contourIndex:
        :param neighbours:
        :return:
        """
        elasticity = []
        curvature = []
        externalEnergy = self.__getExternalEnergyAt__(neighbours)
        for point in neighbours:
            self.contour[contourIndex, 0] = point[0]
            self.contour[contourIndex, 1] = point[1]
            elasticity.append(self.__getElasticityAt__(contourIndex))
            curvature.append(self.__getCurvatureAt__(contourIndex))

        elasticity = np.array(elasticity) / np.max(elasticity)
        curvature = np.array(curvature) / np.max(curvature)
        externalEnergy = np.array(externalEnergy)

        return self.alpha * elasticity + \
               self.beta * curvature + \
               self.gamma * externalEnergy
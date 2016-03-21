import numpy as np
from scipy.signal import convolve2d

class GreedySnake :
    def __init__( self , alpha , beta , gamma , initialContour , image ) :
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.image = np.matrix(image , copy = True , dtype = np.float32)
        self.gradientImage = self.__getEnergyImage__()
        self.contour = np.asarray(initialContour)
        self.movements = 0

    def show( self ) :
        pass

    def __getElasticityAt__( self , contourIndex ) :
        # E(continuity) = || mean(distance) - |P(i) - P(i-1)| ||^2

        dy = self.contour[ : , 0 ] - np.roll(self.contour[ : , 0 ] , -1)
        dx = self.contour[ : , 1 ] - np.roll(self.contour[ : , 1 ] , -1)
        distance = np.sqrt(np.square(dx) + np.square(dy))
        return np.square(np.mean(distance) - distance[ contourIndex ])

    def __getCurvatureAt__( self , contourIdx ) :
        # E(curvature) = || P(i-1) - 2 P(i) + P(i+1) ||^2

        curvature = self.contour[ contourIdx - 1 , : ] - \
                    2 * self.contour[ contourIdx , : ] + \
                    self.contour[ contourIdx + 1 , : ]

        return curvature[ 0 ] ** (2) + curvature[ 1 ] ** (2)

    def __getExternalEnergy__( self ) :
        gradient = [ self.gradientImage[ r , c ] for r , c in self.contour ]

        # Normalize Contour Gradient.
        gradient = (gradient - max(gradient)) / (max(gradient) - min(gradient))
        return sum(gradient)

    def __getEnergyImage__( self ) :
        # E(image) = - || Gaussian * Gradient( Intensity ) ||
        # Apply Gaussian Blurring.
        # Hard-coded Gaussian Kernel. order=3, sigma=1.
        gaussianKernel1D = np.matrix([ 0.27406862 , 0.45186276 , 0.27406862 ])
        gaussianKernel2D = gaussianKernel1D.transpose() * gaussianKernel1D
        gaussianImage = convolve2d(self.image , gaussianKernel2D ,
                                   mode = 'same')

        # Apply Sobel Gradient in both V/H directions.
        sobelHKernel = np.array([ [ -1 , 0 , 1 ] ,
                                  [ -2 , 0 , 2 ] ,
                                  [ -1 , 0 , 1 ] ])
        sobelH = convolve2d(gaussianImage , sobelHKernel , mode = 'same')
        sobelV = convolve2d(gaussianImage , sobelHKernel.transpose() ,
                            mode = 'same')
        return - np.sqrt(np.square(sobelH) + np.square(sobelV))

    def __minimizeCostAt__( self , contourIndex ) :
        # Get the current (r)ow and (c)olumn.
        r = self.contour[ contourIndex , 0 ]
        c = self.contour[ contourIndex , 1 ]

        # Get all neighbors.
        neighbours = [ (r - 1 , c - 1) , (r - 1 , c) , (r - 1 , c + 1) ,
                       (r , c - 1) , (r , c) , (r , c + 1) ,
                       (r + 1 , c - 1) , (r + 1 , c) , (r + 1 , c + 1) ]

        # Calculate cost at each neighbour.
        cost = [ self.__getCostAt__(contourIndex , (row , col)) for row , col in
                 neighbours ]

        # Minimize cost.
        minCost , idx = min((val , idx) for (val , idx) in enumerate(cost))

        # Update the contour to the new neighbor.
        newNeighbour = neighbours[ idx ]
        if newNeighbour != (r , c) :
            self.contour[ contourIndex , 0 ] = newNeighbour[ 0 ]
            self.contour[ contourIndex , 1 ] = newNeighbour[ 1 ]
            self.movements += 1
        else :
            # Else, return the contour to its old state.
            self.contour[ contourIndex , 0 ] = r
            self.contour[ contourIndex , 1 ] = c

    def __getCostAt__( self , contourIndex , neighbour ) :
        # Move contour[ index ] to the neighbour to calculate new cost.
        self.contour[ contourIndex , 0 ] = neighbour[ 0 ]
        self.contour[ contourIndex , 1 ] = neighbour[ 1 ]

        newCost = self.alpha * self.__getElasticityAt__(contourIndex) + \
                  self.beta * self.__getCurvatureAt__(contourIndex) + \
                  self.gamma * self.__getExternalEnergy__()
        return newCost
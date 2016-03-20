import numpy as np





def myHoughCircles( binaryImage , threshold ) :
    maxRadius = int(min( binaryImage.shape )/2 + 0.5)
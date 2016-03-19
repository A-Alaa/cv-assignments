import numpy as np


def frange( initial , final , step ) :
    while initial < final :
        yield initial
        initial += step


def __houghVoteLines__( point , maxRadius , accumulator , cosines , sines ) :
    for angle_index in range(0 , cosines.shape[0] - 1) :
        rho = point[0] * cosines[angle_index] + point[1] * sines[angle_index]
        accumulator[angle_index , int(rho + maxRadius)] += 1


def __extractLines__( accumulator , angles , threshold ) :
    lines = []
    thresholdActual = threshold * accumulator.max()

    for angle_index in range(0 , accumulator.shape[0] - 1) :
        for rho in range(0 , accumulator.shape[1] - 1) :
            if accumulator[angle_index , rho] > thresholdActual :
                angle = angles[angle_index]
                # print (angle , rho , accumulator[angle_index , rho])
                lines.append((angle , rho))

    return lines


def myHoughLines( binaryImage , angleAccuracy , threshold ) :
    angles = frange(0 , np.pi , angleAccuracy)
    angles = np.fromiter(angles , dtype = np.float16)

    cartesianWidth = binaryImage.shape[0]
    cartesianHeight = binaryImage.shape[1]

    center = (cartesianWidth / 2.0 , cartesianHeight / 2.0)
    maxRadius = np.sqrt(2) * max(cartesianWidth , cartesianHeight) / 2.0

    polarWidth = 180  # 180 degree
    polarHeight = 2 * int(maxRadius)

    # Construct the empty accumulator
    accumulator = np.empty((polarWidth , polarHeight) , dtype = np.uint32)
    print "Accumulator size:"
    print accumulator.shape
    # Caculate all sines and cosines once to reduce the overhead of redundant
    # float operations.
    sines = np.sin(angles , dtype = np.float16)
    cosines = np.cos(angles , dtype = np.float16)

    for row in range(0 , binaryImage.shape[0] - 1) :
        for col in range(0 , binaryImage.shape[1] - 1) :
            if binaryImage[row , col] != 0 :
                __houghVoteLines__((row - center[0] , col - center[1]) ,
                                   maxRadius , accumulator , cosines , sines)


    lines = __extractLines__(accumulator , angles , threshold)

    print("lines count:")
    print len(lines)
    return lines

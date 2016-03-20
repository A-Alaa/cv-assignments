import numpy as np


def myHoughCircles( binaryImage , threshold ) :
    maxRadius = int(min(binaryImage.shape) / 2 + 0.5)
    cartesianWidth = binaryImage.shape[ 1 ]
    cartesianHeight = binaryImage.shape[ 0 ]

    accumulator = np.ndarray((cartesianHeight , cartesianWidth , maxRadius) ,
                             dtype = int )
    accumulator.fill(0)

    # idx = np.array([2,2,2,2,2,2], dtype  = int )
    # v = np.array([1,2,3,4,5,6,7])
    # v[ idx ] += 1
    # print v
    # exit(1)
    print "Extracting edge points"
    edge_points = [ (row , col) for row in range(binaryImage.shape[ 0 ]) for col
                    in range(binaryImage.shape[ 1 ]) if
                    binaryImage[ row , col ] != 0 ]

    edge_points = np.asanyarray(edge_points , dtype = np.uint16)

    print "Accumulation"
    for row in range(accumulator.shape[ 0 ]) :
        for col in range(accumulator.shape[ 1 ]) :
            radius = np.round(np.sqrt(np.square(edge_points[ : , 0 ] - row) +
                                      np.square(edge_points[ : , 1 ] - col)))
            radius = radius.astype(dtype = int , copy = False)

            radius , count = np.unique(radius , return_counts = True)
            # count = count.astype(dtype = np.uint16 , copy = False)
            accumulator[ row , col , radius ] += count


    print "Max circle"
    print np.amax(accumulator)
    print "Extract most candidate circles"
    circles = np.where(accumulator > threshold)
    circles = np.transpose(circles)
    circles = tuple(map(tuple , circles))
    circles = [ (col , row , r) for row , col , r in circles ]

    return circles
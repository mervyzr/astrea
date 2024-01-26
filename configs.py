import numpy as np


##############################################################################

config = "sod"
cells = 100
cfl = .5
gamma = 1.4
solver = "ppm"

runType = "single"
livePlot = True

saveFile = False
snapshots = 1
makeVideo = False


if config == "sod":
    startPos = 0
    endPos = 1
    shockPos = .5
    tEnd = .2
    boundary = "outflow"

    initialLeft = np.array([1,0,0,0,1])  # primitive variables
    initialRight = np.array([.125,0,0,0,.1])  # primitive variables
elif config == "sin":
    startPos = 0
    endPos = 1
    shockPos = 1
    tEnd = 2
    boundary = "periodic"

    initialLeft = np.array([0,1,1,1,1])  # primitive variables
    initialRight = np.array([0,0,0,0,0])  # primitive variables
elif config == "sedov":
    startPos = -10
    endPos = 10
    shockPos = 1
    tEnd = .6
    boundary = "outflow"

    initialLeft = np.array([1,0,0,0,100])  # primitive variables
    initialRight = np.array([1,0,0,0,1])  # primitive variables
else:
    startPos = 0
    endPos = 1
    shockPos = .5
    tEnd = .2
    boundary = "outflow"

    initialLeft = np.array([1,0,0,0,1])  # primitive variables
    initialRight = np.array([.125,0,0,0,.1])  # primitive variables
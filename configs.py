

##############################################################################

config = "sod"
cells = 100
cfl = .8
gamma = 1.4
solver = "linear"

runType = "single"
livePlot = True

saveFile = False
snapshots = 1
makeVideo = False

if config == "sin":
    # sin-wave
    startPos = 0
    endPos = 1
    shockPos = 1
    tEnd = 2
elif config == "sedov":
    # sedov shock
    startPos = -10
    endPos = 10
    shockPos = 1
    tEnd = .6
else:
    # sod shock
    startPos = 0
    endPos = 1
    shockPos = .5
    tEnd = .2

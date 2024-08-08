import numpy as np

##############################################################################
# Initial conditions for test configs
##############################################################################

def generateTestConditions(config):
    if config == "sod":
        startPos = 0
        endPos = 1
        shockPos = .5
        tEnd = .2
        boundary = "edge"  # outflow
        initialLeft = np.array([1,0,0,0,1,0,0,0])  # primitive variables [rho, vx, vy, vz, P, Bx, By, Bz]
        initialRight = np.array([.125,0,0,0,.1,0,0,0])  # primitive variables [rho, vx, vy, vz, P, Bx, By, Bz]
        misc = None

    elif config == "sin":
        startPos = 0
        endPos = 1
        shockPos = 1
        tEnd = 1
        boundary = "wrap"  # periodic
        initialLeft = np.array([0,1,1,1,1,0,0,0])
        initialRight = np.array([0,1,1,1,1,0,0,0])
        misc = {'freq':2, 'ampl':.1, 'y_offset':1}

    elif config == "sinc":
        startPos = -4
        endPos = 4
        shockPos = 0
        tEnd = 20
        boundary = "edge"  # periodic
        initialLeft = np.array([0,1e-6,1e-6,1e-6,1e-3,0,0,0])
        initialRight = np.array([0,1e-6,1e-6,1e-6,1e-3,0,0,0])
        misc = {'freq':10, 'ampl':1, 'y_offset':1}

    elif config == "sedov":
        startPos = -10
        endPos = 10
        shockPos = .5  # blast boundary
        tEnd = .6
        boundary = "edge"  # outflow
        initialLeft = np.array([1,0,0,0,100,0,0,0])
        initialRight = np.array([1,0,0,0,1,0,0,0])
        misc = None

    elif "shu" in config or "osher" in config:
        startPos = -1
        endPos = 1
        shockPos = -.8
        tEnd = .47
        boundary = "edge"  # outflow
        initialLeft = np.array([3.857143,2.629369,0,0,10.3333,0,0,0])
        initialRight = np.array([0,0,0,0,1,0,0,0])
        misc = {'freq':5, 'ampl':.2, 'y_offset':1}

    elif config.startswith('gauss'):
        startPos = 0
        endPos = 1
        shockPos = 1
        tEnd = 1
        boundary = "wrap"  # periodic
        initialLeft = np.array([0,1,1,1,1e-6,0,0,0])
        initialRight = np.array([0,1,1,1,1e-6,0,0,0])
        misc = {'ampl':.9999, 'fwhm':.01, 'y_offset':1}

    elif config.startswith('sq'):
        startPos = -1
        endPos = 1
        shockPos = 1/3
        tEnd = .05
        boundary = "wrap"  # periodic
        initialLeft = np.array([1,1,0,0,1,0,0,0])
        initialRight = np.array([.01,1,0,0,1,0,0,0])
        misc = None

    elif "toro" in config:
        startPos = 0
        endPos = 1
        boundary = "edge"  # outflow
        misc = None

        if "2" in config:
            shockPos = .5
            tEnd = .14
            initialLeft = np.array([1,-2,0,0,.4,0,0,0])
            initialRight = np.array([1,2,0,0,.4,0,0,0])

        elif "3" in config:
            shockPos = .5
            tEnd = .012
            initialLeft = np.array([1,0,0,0,1000,0,0,0])
            initialRight = np.array([1,0,0,0,.01,0,0,0])

        elif "4" in config:
            shockPos = .3
            tEnd = .05
            initialLeft = np.array([5.99924,19.5975,0,0,460.894,0,0,0])
            initialRight = np.array([5.99242,-6.19633,0,0,46.095,0,0,0])

        elif "5" in config:
            shockPos = .8
            tEnd = .012
            initialLeft = np.array([1,-19.59745,0,0,1000,0,0,0])
            initialRight = np.array([1,-19.59745,0,0,.01,0,0,0])

        else:
            shockPos = .3
            tEnd = .2
            initialLeft = np.array([1,.75,0,0,1,0,0,0])
            initialRight = np.array([.125,0,0,0,.1,0,0,0])

    elif "ryu" in config or "jones" in config or "rj" in config:
        startPos = -.5
        endPos = .5
        shockPos = 0
        tEnd = .15
        boundary = "edge"  # outflow
        initialLeft = np.array([1.08,1.2,.01,.5,.95,.5641895835477562,1.0155412503859613,.5641895835477562])
        initialRight = np.array([1,0,0,0,1,.5641895835477562,1.1283791670955125,.5641895835477562])
        misc = None

    else:
        startPos = 0
        endPos = 1
        shockPos = .5
        tEnd = .2
        boundary = "edge"  # outflow
        initialLeft = np.array([1,0,0,0,1,0,0,0])
        initialRight = np.array([.125,0,0,0,.1,0,0,0])
        misc = None

    return {'startPos':startPos, 'endPos':endPos, 'shockPos':shockPos, 'tEnd':tEnd, 'boundary':boundary.lower(), 'misc':misc, 'initialLeft':initialLeft, 'initialRight':initialRight}
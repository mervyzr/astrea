import numpy as np

##############################################################################
# Initial conditions for test configs
##############################################################################

def generate_test_conditions(config):
    if config == "sod":
        start_pos = 0
        end_pos = 1
        shock_pos = .5
        t_end = .2
        boundary = "edge"  # outflow
        initial_left = np.array([1,0,0,0,1,0,0,0])  # primitive variables [rho, vx, vy, vz, P, Bx, By, Bz]
        initial_right = np.array([.125,0,0,0,.1,0,0,0])  # primitive variables [rho, vx, vy, vz, P, Bx, By, Bz]
        misc = None

    elif config == "sin":
        start_pos = 0
        end_pos = 1
        shock_pos = 1
        t_end = 1
        boundary = "wrap"  # periodic
        initial_left = np.array([0,1,1,0,1,0,0,0])
        initial_right = np.array([0,1,1,0,1,0,0,0])
        misc = {'freq':2, 'ampl':.1, 'y_offset':1}

    elif config == "sedov":
        start_pos = -10
        end_pos = 10
        shock_pos = .5  # blast boundary
        t_end = .6
        boundary = "wrap"  # periodic
        initial_left = np.array([1,0,0,0,100,0,0,0])
        initial_right = np.array([1,0,0,0,1,0,0,0])
        misc = None

    elif "shu" in config or "osher" in config:
        start_pos = -1
        end_pos = 1
        shock_pos = -.8
        t_end = .47
        boundary = "edge"  # outflow
        initial_left = np.array([3.857143,2.629369,0,0,10.3333,0,0,0])
        initial_right = np.array([0,0,0,0,1,0,0,0])
        misc = {'freq':5, 'ampl':.2, 'y_offset':1}

    elif config.startswith('gauss'):
        start_pos = -1
        end_pos = 1
        shock_pos = 1
        t_end = 2
        boundary = "wrap"  # periodic
        initial_left = np.array([0,1,1,0,1e-6,0,0,0])
        initial_right = np.array([0,1,1,0,1e-6,0,0,0])
        misc = {'ampl':.9999, 'fwhm':.02, 'y_offset':1}

    elif config.startswith('sq'):
        start_pos = -1
        end_pos = 1
        shock_pos = 1/3
        t_end = .05
        boundary = "wrap"  # periodic
        initial_left = np.array([1,1,0,0,1,0,0,0])
        initial_right = np.array([.01,1,0,0,1,0,0,0])
        misc = None

    elif "ryu" in config or "jones" in config or "rj" in config:
        start_pos = -.5
        end_pos = .5
        shock_pos = 0
        t_end = .15
        boundary = "edge"  # outflow
        initial_left = np.array([1.08,1.2,.01,.5,.95,.5641895835477562,1.0155412503859613,.5641895835477562])
        initial_right = np.array([1,0,0,0,1,.5641895835477562,1.1283791670955125,.5641895835477562])
        misc = None

    elif "brio" in config or "wu" in config or "bw" in config:
        start_pos = -.5
        end_pos = .5
        shock_pos = 0
        t_end = .1
        boundary = "edge"  # outflow
        initial_left = np.array([1,0,0,0,1,.75,1,0])
        initial_right = np.array([.125,0,0,0,.1,.75,-1,0])
        misc = None

    elif config == "khi" or config == "kelvin-helmholtz" or ("kelvin" in config or "helmholtz" in config):
        start_pos = -1
        end_pos = 1
        shock_pos = .5
        t_end = 1
        boundary = "wrap"  # periodic
        initial_left = np.array([2,-.5,0,0,2.5,0,0,0])
        initial_right = np.array([1,.5,0,0,2.5,0,0,0])
        misc = {'perturb_ampl':.01, 'freq':2}

    elif "toro" in config:
        start_pos = 0
        end_pos = 1
        boundary = "edge"  # outflow
        misc = None

        if "2" in config:
            shock_pos = .5
            t_end = .14
            initial_left = np.array([1,-2,0,0,.4,0,0,0])
            initial_right = np.array([1,2,0,0,.4,0,0,0])

        elif "3" in config:
            shock_pos = .5
            t_end = .012
            initial_left = np.array([1,0,0,0,1000,0,0,0])
            initial_right = np.array([1,0,0,0,.01,0,0,0])

        elif "4" in config:
            shock_pos = .3
            t_end = .05
            initial_left = np.array([5.99924,19.5975,0,0,460.894,0,0,0])
            initial_right = np.array([5.99242,-6.19633,0,0,46.095,0,0,0])

        elif "5" in config:
            shock_pos = .8
            t_end = .012
            initial_left = np.array([1,-19.59745,0,0,1000,0,0,0])
            initial_right = np.array([1,-19.59745,0,0,.01,0,0,0])

        else:
            shock_pos = .3
            t_end = .2
            initial_left = np.array([1,.75,0,0,1,0,0,0])
            initial_right = np.array([.125,0,0,0,.1,0,0,0])

    elif "ll" in config or "lax-liu" in config:
        start_pos = 0
        end_pos = 1
        shock_pos = .5
        boundary = "wrap"  # periodic

        if "3" in config:
            t_end = .3
            initial_left = np.array([.5323,1.206,0,0,.3,0,0,0])
            initial_right = np.array([1.5,0,0,0,1.5,0,0,0])
            misc = {'bottom_left':np.array([.138,1.206,1.206,0,.029,0,0,0]), 'bottom_right':np.array([.5323,0,1.206,0,.3,0,0,0])}

        elif "4" in config:
            t_end = .25
            initial_left = np.array([.5065,.8939,0,0,.35,0,0,0])
            initial_right = np.array([1.1,0,0,0,1.1,0,0,0])
            misc = {'bottom_left':np.array([1.1,.8939,.8939,0,1.1,0,0,0]), 'bottom_right':np.array([.5065,0,.8939,0,.35,0,0,0])}

        elif "6" in config:
            t_end = .3
            initial_left = np.array([2,.75,.5,0,1,0,0,0])
            initial_right = np.array([1,.75,-.5,0,1,0,0,0])
            misc = {'bottom_left':np.array([1,-.75,.5,0,1,0,0,0]), 'bottom_right':np.array([3,-.75,-.5,0,1,0,0,0])}

        elif "11" in config:
            t_end = .3
            initial_left = np.array([.5313,.8276,0,0,.4,0,0,0])
            initial_right = np.array([1,.1,0,0,1,0,0,0])
            misc = {'bottom_left':np.array([.8,.1,0,0,.4,0,0,0]), 'bottom_right':np.array([.5313,.1,.7276,0,.4,0,0,0])}

        elif "15" in config:
            t_end = .2
            initial_left = np.array([.5197,-.6259,-.3,0,.4,0,0,0])
            initial_right = np.array([1,.1,-.3,0,1,0,0,0])
            misc = {'bottom_left':np.array([.8,.1,-.3,0,.4,0,0,0]), 'bottom_right':np.array([.5313,.1,.4276,0,.4,0,0,0])}

        else:
            t_end = .25
            initial_left = np.array([1,.7276,0,0,1,0,0,0])
            initial_right = np.array([.5313,0,0,0,.4,0,0,0])
            misc = {'bottom_left':np.array([.8,0,0,0,1,0,0,0]), 'bottom_right':np.array([1,0,.7276,0,1,0,0,0])}

    else:
        start_pos = 0
        end_pos = 1
        shock_pos = .5
        t_end = .2
        boundary = "edge"  # outflow
        initial_left = np.array([1,0,0,0,1,0,0,0])
        initial_right = np.array([.125,0,0,0,.1,0,0,0])
        misc = None

    return {'start_pos':start_pos, 'end_pos':end_pos, 'shock_pos':shock_pos, 't_end':t_end, 'boundary':boundary.lower(), 'misc':misc, 'initial_left':initial_left, 'initial_right':initial_right}
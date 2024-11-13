import numpy as np

##############################################################################
# Initial conditions for test configs
##############################################################################

def generate_test_conditions(config, cells):
    # [Sod, 1978]
    if "sod" in config:
        start_pos = 0
        end_pos = 1
        shock_pos = .5
        t_end = .2
        boundary = "edge"  # outflow
        initial_left = np.array([1,0,0,0,1,0,0,0])  # primitive variables [rho, vx, vy, vz, P, Bx, By, Bz]
        initial_right = np.array([.125,0,0,0,.1,0,0,0])  # primitive variables [rho, vx, vy, vz, P, Bx, By, Bz]
        misc = None

    # [Sedov, 1959]
    elif "sedov" in config:
        start_pos = -10
        end_pos = 10
        shock_pos = .5  # blast boundary
        t_end = .6
        boundary = "wrap"  # periodic
        initial_left = np.array([1,0,0,0,100,0,0,0])
        initial_right = np.array([1,0,0,0,1,0,0,0])
        misc = None

    # [Shu & Osher, 1989]
    elif "shu" in config or "osher" in config or config == "so":
        start_pos = -1
        end_pos = 1
        shock_pos = -.8
        t_end = .47
        boundary = "edge"  # outflow
        initial_left = np.array([3.857143,2.629369,0,0,10.3333,0,0,0])
        initial_right = np.array([0,0,0,0,1,0,0,0])
        misc = {'freq':5, 'ampl':.2, 'y_offset':1}

    elif config.startswith("sin"):
        start_pos = 0
        end_pos = 1
        shock_pos = 1
        t_end = 1
        boundary = "wrap"  # periodic
        initial_left = np.array([0,1,1,0,1,0,0,0])
        initial_right = np.array([0,1,1,0,1,0,0,0])
        misc = {'freq':2, 'ampl':.1, 'y_offset':2}

    elif config.startswith('gauss'):
        start_pos = -1
        end_pos = 1
        shock_pos = 1
        t_end = 2
        boundary = "wrap"  # periodic
        initial_left = np.array([0,1,1,0,1e-6,0,0,0])
        initial_right = np.array([0,1,1,0,1e-6,0,0,0])
        misc = {'peak_pos':0, 'ampl':.75, 'fwhm':.08, 'y_offset':1}

    elif "slow" in config:
        start_pos = 0
        end_pos = 1
        shock_pos = .5
        t_end = .08
        boundary = "edge"  # outflow
        initial_left = np.array([5.6698,-1.5336,0,0,100,0,0,0])
        initial_right = np.array([1,-10.5636,0,0,1,0,0,0])
        misc = None

    elif config.startswith('sq'):
        start_pos = -1
        end_pos = 1
        shock_pos = 1/3
        t_end = .05
        boundary = "wrap"  # periodic
        initial_left = np.array([1,1,0,0,1,0,0,0])
        initial_right = np.array([.01,1,0,0,1,0,0,0])
        misc = None

    # [Ryu & Jones, 1995]
    elif "ryu" in config or "jones" in config or config == "rj":
        start_pos = -.5
        end_pos = .5
        shock_pos = 0
        t_end = .15
        boundary = "edge"  # outflow
        initial_left = np.array([1.08,1.2,.01,.5,.95,.5641895835477562,1.0155412503859613,.5641895835477562])
        initial_right = np.array([1,0,0,0,1,.5641895835477562,1.1283791670955125,.5641895835477562])
        misc = None

    # [Brio & Wu, 1988]
    elif "brio" in config or "wu" in config or config == "bw":
        start_pos = -.5
        end_pos = .5
        shock_pos = 0
        t_end = .1
        boundary = "edge"  # outflow
        initial_left = np.array([1,0,0,0,1,.75,1,0])
        initial_right = np.array([.125,0,0,0,.1,.75,-1,0])
        misc = None

    elif "kelvin" in config or "helmholtz" in config or config == "khi":
        start_pos = 0
        end_pos = 1
        shock_pos = .5
        t_end = 2
        boundary = "wrap"  # periodic
        initial_left = np.array([2,-.5,0,0,2.5,0,0,0])
        initial_right = np.array([1,.5,0,0,2.5,0,0,0])
        misc = {'perturb_ampl':.5, 'freq':4}

    # [Yee et. al., 1999]
    elif "isentropic" in config or "vortex" in config or config == "ivc":
        start_pos = 0
        end_pos = 10
        shock_pos = 5
        t_end = 1
        boundary = "wrap"  # periodic
        initial_left = np.array([1,0,0,0,1,0,0,0])
        initial_right = np.array([1,0,0,0,1,0,0,0])
        misc = {'vortex_str':5, 'freq':2}

    # [Toro, 1999, p.225]
    elif "toro" in config:
        start_pos = 0
        end_pos = 1
        boundary = "edge"  # outflow
        misc = None

        # Double rarefaction wave
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

    # [Lax & Liu, 1998]
    elif ("lax" in config or "liu" in config) or "ll" in config:
        start_pos = 0
        end_pos = 1
        shock_pos = .5
        t_end = 2
        boundary = "wrap"  # periodic

        if "ll" in config:
            index = int(config.replace(' ','').split('ll')[-1])
        else:
            index = int(config.replace(' ','').split('liu')[-1])

        if index in [1, 2]:
            initial_left = np.array([.5197,-.7259,0,0,.4,0,0,0])
            initial_right = np.array([1,0,0,0,1,0,0,0])
            if index == 1:
                misc = {'bottom_left':np.array([.1072,-.7259,-1.4045,0,.0439,0,0,0]), 'bottom_right':np.array([.2579,0,-1.4045,0,.15,0,0,0])}
            else:
                misc = {'bottom_left':np.array([1,-.7259,-.7259,0,1,0,0,0]), 'bottom_right':np.array([.5197,0,-.7259,0,.4,0,0,0])}

        elif index == 3:
            initial_left = np.array([1.5,0,0,0,1.5,0,0,0])
            initial_right = np.array([.5323,1.206,0,0,.3,0,0,0])
            misc = {'bottom_left':np.array([.5323,0,1.206,0,.3,0,0,0]), 'bottom_right':np.array([.138,1.206,1.206,0,.029,0,0,0])}

        elif index == 4:
            initial_left = np.array([.5065,.8939,0,0,.35,0,0,0])
            initial_right = np.array([1.1,0,0,0,1.1,0,0,0])
            misc = {'bottom_left':np.array([1.1,.8939,.8939,0,1.1,0,0,0]), 'bottom_right':np.array([.5065,0,.8939,0,.35,0,0,0])}

        elif index in [5, 6]:
            coeff = -1**index
            initial_left = np.array([2,coeff*.75,.5,0,1,0,0,0])
            initial_right = np.array([1,coeff*.75,-.5,0,1,0,0,0])
            misc = {'bottom_left':np.array([1,-coeff*.75,.5,0,1,0,0,0]), 'bottom_right':np.array([3,-coeff*.75,-.5,0,1,0,0,0])}

        elif index == 7:
            initial_left = np.array([.5197,-.6259,.1,0,.4,0,0,0])
            initial_right = np.array([1,.1,.1,0,1,0,0,0])
            misc = {'bottom_left':np.array([.8,.1,.1,0,.4,0,0,0]), 'bottom_right':np.array([.5197,.1,-.6259,0,.4,0,0,0])}

        elif index == 8:
            initial_left = np.array([1,-.6259,.1,0,1,0,0,0])
            initial_right = np.array([.5197,.1,.1,0,.4,0,0,0])
            misc = {'bottom_left':np.array([.8,.1,.1,0,1,0,0,0]), 'bottom_right':np.array([1,.1,-.6259,0,1,0,0,0])}

        elif index == 9:
            initial_left = np.array([2,0,-.3,0,1,0,0,0])
            initial_right = np.array([1,0,.3,0,1,0,0,0])
            misc = {'bottom_left':np.array([1.039,0,-.8133,0,.4,0,0,0]), 'bottom_right':np.array([.5197,0,-.4259,0,.4,0,0,0])}

        elif index == 10:
            initial_left = np.array([.5,0,.6076,0,1,0,0,0])
            initial_right = np.array([1,0,.4297,0,1,0,0,0])
            misc = {'bottom_left':np.array([.2281,0,-.6076,0,.3333,0,0,0]), 'bottom_right':np.array([.4562,0,-.4297,0,.3333,0,0,0])}

        elif index == 11:
            initial_left = np.array([.5313,.8276,0,0,.4,0,0,0])
            initial_right = np.array([1,.1,0,0,1,0,0,0])
            misc = {'bottom_left':np.array([.8,.1,0,0,.4,0,0,0]), 'bottom_right':np.array([.5313,.1,.7276,0,.4,0,0,0])}

        elif index == 12:
            initial_left = np.array([1,.7276,0,0,1,0,0,0])
            initial_right = np.array([.5313,0,0,0,.4,0,0,0])
            misc = {'bottom_left':np.array([.8,0,0,0,1,0,0,0]), 'bottom_right':np.array([1,0,.7276,0,1,0,0,0])}

        elif index == 13:
            initial_left = np.array([2,.3,0,0,1,0,0,0])
            initial_right = np.array([1,0,-.3,0,1,0,0,0])
            misc = {'bottom_left':np.array([1.0625,0,.8145,0,.4,0,0,0]), 'bottom_right':np.array([.5313,0,.4276,0,.4,0,0,0])}

        elif index == 14:
            initial_left = np.array([1,0,-1.2172,0,8,0,0,0])
            initial_right = np.array([2,0,-.5606,0,8,0,0,0])
            misc = {'bottom_left':np.array([.4736,0,1.2172,0,2.6667,0,0,0]), 'bottom_right':np.array([.9474,0,1.1606,0,2.6667,0,0,0])}

        elif index == 15:
            initial_left = np.array([.5197,-.6259,-.3,0,.4,0,0,0])
            initial_right = np.array([1,.1,-.3,0,1,0,0,0])
            misc = {'bottom_left':np.array([.8,.1,-.3,0,.4,0,0,0]), 'bottom_right':np.array([.5313,.1,.4276,0,.4,0,0,0])}

        elif index == 16:
            initial_left = np.array([1.0222,-.6179,.1,0,1,0,0,0])
            initial_right = np.array([.5313,.1,.1,0,.4,0,0,0])
            misc = {'bottom_left':np.array([.8,.1,.1,0,1,0,0,0]), 'bottom_right':np.array([1,.1,.8276,0,1,0,0,0])}

        elif index in [17, 18, 19]:
            if index == 17:
                v1, v4 = -.4, -1.1259
            elif index == 18:
                v1, v4 = 1, .2741
            else:
                v1, v4 = .3, -.4259
            initial_left = np.array([2,0,-.3,0,1,0,0,0])
            initial_right = np.array([1,0,v1,0,1,0,0,0])
            misc = {'bottom_left':np.array([1.0625,0,.2145,0,.4,0,0,0]), 'bottom_right':np.array([.5197,0,v4,0,.4,0,0,0])}

    else:
        start_pos = 0
        end_pos = 1
        shock_pos = .5
        t_end = .2
        boundary = "edge"  # outflow
        initial_left = np.array([1,0,0,0,1,0,0,0])
        initial_right = np.array([.125,0,0,0,.1,0,0,0])
        misc = None

    return {'start_pos':start_pos, 'end_pos':end_pos, 'shock_pos':shock_pos, 't_end':t_end, 'boundary':boundary.lower(), 'misc':misc, 'initial_left':initial_left, 'initial_right':initial_right, 'dx':abs(end_pos-start_pos)/cells}
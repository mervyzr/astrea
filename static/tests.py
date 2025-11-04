import numpy as np

##############################################################################
# Initial conditions for test configs
##############################################################################

# Primitive variables [rho, vx, vy, vz, P, Bx, By, Bz]
def generate_test_conditions(config_variables):
    config, cells, gamma = config_variables['config'], config_variables['cells'], config_variables['gamma']

    # [Sod, 1978]
    if "sod" in config:
        axis_coord = [0,1]
        shock_pos = .5
        t_end = .2
        boundary = "edge"  # outflow
        initial_left = np.array([1,0,0,0,1,0,0,0])
        initial_right = np.array([.125,0,0,0,.1,0,0,0])
        misc = None

    # [Sedov, 1959]
    elif "sedov" in config:
        axis_coord = [-10,10]
        shock_pos = .5
        t_end = 1
        boundary = "wrap"  # periodic
        initial_left = np.array([1,0,0,0,100,0,0,0])
        initial_right = np.array([1,0,0,0,1e-12,0,0,0])
        misc = None

    # [Shu & Osher, 1989]
    elif "shu" in config or "osher" in config or config == "so":
        axis_coord = [-1,1]
        shock_pos = -.8
        t_end = .47
        boundary = "edge"
        initial_left = np.array([3.857143,2.629369,0,0,10.3333,0,0,0])
        initial_right = np.array([0,0,0,0,1,0,0,0])
        misc = {'freq':5, 'ampl':.2, 'y_offset':1}

    elif config.startswith("sin"):
        axis_coord = [0,1]
        shock_pos = 1
        t_end = 1
        boundary = "wrap"
        initial_left = np.array([0,1,1,1,1,0,0,0])
        initial_right = np.array([0,1,1,1,1,0,0,0])
        misc = {'freq':2, 'ampl':.1, 'y_offset':2}

    elif config.startswith('gauss'):
        axis_coord = [-1,1]
        shock_pos = 1
        t_end = 2
        boundary = "wrap"
        initial_left = np.array([0,1,1,1,1e-6,0,0,0])
        initial_right = np.array([0,1,1,1,1e-6,0,0,0])
        misc = {'peak_pos':0, 'ampl':.75, 'fwhm':.08, 'y_offset':1}

    elif "noh" in config:
        axis_coord = [0,1]
        shock_pos = .1
        t_end = 1
        boundary = "edge"
        initial_left = np.array([1,0,0,0,1e-6,0,0,0])
        initial_right = np.array([16,0,0,0,16/3,0,0,0])
        misc = None

    elif "slow" in config:
        axis_coord = [0,1]
        shock_pos = .5
        t_end = .08
        boundary = "edge"
        initial_left = np.array([5.6698,-1.5336,0,0,100,0,0,0])
        initial_right = np.array([1,-10.5636,0,0,1,0,0,0])
        misc = None

    elif config.startswith('sq'):
        axis_coord = [-1,1]
        shock_pos = 1/3
        t_end = .05
        boundary = "wrap"
        initial_left = np.array([1,1,0,0,1,0,0,0])
        initial_right = np.array([.01,1,0,0,1,0,0,0])
        misc = None

    # [Ryu & Jones, 1995]
    elif "ryu" in config or "jones" in config or config == "rj":
        axis_coord = [-.5,.5]
        shock_pos = 0
        t_end = .15
        boundary = "edge"
        initial_left = np.array([1.08,1.2,.01,.5,.95,1/np.sqrt(np.pi),1.8/np.sqrt(np.pi),1/np.sqrt(np.pi)])
        initial_right = np.array([1,0,0,0,1,1/np.sqrt(np.pi),2/np.sqrt(np.pi),1/np.sqrt(np.pi)])
        misc = None

    # [Brio & Wu, 1988]
    elif "brio" in config or "wu" in config or config == "bw":
        axis_coord = [-.5,.5]
        shock_pos = 0
        t_end = .1
        boundary = "edge"
        initial_left = np.array([1,0,0,0,1,.75,1,0])
        initial_right = np.array([.125,0,0,0,.1,.75,-1,0])
        misc = None

    elif "kelvin" in config or "helmholtz" in config or "khi" in config:
        axis_coord = [-1,1]
        shock_pos = .4
        t_end = 5
        boundary = "wrap"
        initial_left = np.array([2,.5,0,0,2.5,0,0,0])
        initial_right = np.array([1,-.5,0,0,2.5,0,0,0])
        misc = {'perturb_ampl':.05, 'ampl':.25, 'freq':4, 'Bx':np.sqrt(np.pi)}

    elif "turb" in config:
        axis_coord = [0,1]
        shock_pos = .5
        t_end = 1
        boundary = "wrap"
        initial_left = np.array([1,1e-6,1e-6,1e-6,1/gamma,np.sqrt(2),0,1e-6])
        initial_right = np.array([1,1e-6,1e-6,1e-6,1/gamma,np.sqrt(2),0,1e-6])
        misc = {'force':'solenoidal', 'mach':5, 'beta':1, 'ampl':.5, 'k1':1, 'k2':2, 'pk':0}

    # [Pang & Wu, 2025]
    elif config in ["ivc", "isentropic"]:
        axis_coord = [0,10]
        shock_pos = 5
        t_end = 10
        boundary = "wrap"
        initial_left = np.array([1,0,0,0,1,0,0,0])
        initial_right = np.array([1,0,0,0,1,0,0,0])
        misc = {'vortex_str':1, 'freq':2}

    # [Gresho & Chan, 1990]
    elif "gresho" in config:
        axis_coord = [0,1]
        shock_pos = .5
        t_end = 1
        boundary = "wrap"
        initial_left = np.array([1,0,0,0,0,0,0,0])
        initial_right = np.array([1,0,0,0,0,0,0,0])
        misc = {'mach':.1}

    # [Orszag & Tang, 1998; Stone et al., 2008]
    elif "orszag" in config or "tang" in config or config == "ot":
        axis_coord = [0,1]
        shock_pos = 0
        t_end = .8
        boundary = "wrap"
        initial_left = np.array([gamma*2,0,0,0,gamma,0,0,0])
        initial_right = np.array([gamma*2,0,0,0,gamma,0,0,0])
        misc = {'ampl':1}

    # [Balsara, 2004; Li, 2010]
    elif "vortex" in config and config.startswith("mhd"):
        boundary = "wrap"
        t_end = 20

        if len(cells) < 3:
            axis_coord = [-10,10]
            shock_pos = 10
            initial_left = np.array([1,0,0,0,1,0,0,0])
            initial_right = np.array([1,0,0,0,1,0,0,0])
            misc = {'kappa':5, 'mu':5}
        else:
            axis_coord = [-5,5]
            shock_pos = 5
            initial_left = np.array([1,0,0,2,1,0,0,0])
            initial_right = np.array([1,0,0,2,1,0,0,0])
            misc = {'kappa':1/np.sqrt(2*np.pi), 'mu':1/np.sqrt(2*np.pi), 'q':1}

    # [Balsara & Spicer, 1999]
    elif "rotor" in config:
        axis_coord = [-.5,.5]
        shock_pos = .1
        t_end = .2
        boundary = "wrap"
        initial_left = np.array([10,0,0,0,1,5/np.sqrt(4*np.pi),0,0])
        initial_right = np.array([1,0,0,0,1,5/np.sqrt(4*np.pi),0,0])
        misc = {'omega':20, 'ring_pos':.115}

    # [Felker & Stone, 2018]
    elif "blast" in config and config.startswith("mhd"):
        axis_coord = [-.5,.5]
        shock_pos = .1
        t_end = .2
        boundary = "wrap"
        initial_left = np.array([1,0,0,0,10,0,0,0])
        initial_right = np.array([1,0,0,0,.1,0,0,0])
        misc = {'ampl':1/np.sqrt(2)}

    # [Gardiner & Stone, 2005]
    elif "sheet" in config or "current" in config:
        axis_coord = [-.5,.5]
        shock_pos = .25
        t_end = 10
        boundary = "wrap"
        initial_left = np.array([1,0,0,0,.05/(4*np.pi),0,1/np.sqrt(4*np.pi),0])
        initial_right = np.array([1,0,0,0,.05/(4*np.pi),0,1/np.sqrt(4*np.pi),0])
        misc = {'ampl':.1}

    # [Dai & Woodward, 1998]
    elif "cloud" in config:
        axis_coord = [0,1]
        shock_pos = .6
        t_end = 1
        boundary = "edge"
        initial_left = np.array([3.86859,0,0,0,167.345,0,2.1826182,-2.1826182])
        initial_right = np.array([1,-11.2536,0,0,1,0,.56418958,.56418958])
        misc = None

    # [Wu & Shu, 2018]
    elif "jet" in config:
        axis_coord = [-.5,.5]
        shock_pos = -.49
        t_end = .01
        boundary = "edge"
        initial_left = np.array([gamma*.1,0,0,0,1,0,np.sqrt(2000),0])
        initial_right = np.array([gamma*.1,0,0,0,1,0,np.sqrt(2000),0])
        misc = None

    # [Ziegler, 2000]
    elif "circular" in config or "polarised" in config or "alfven" in config or config == "cpaw":
        axis_coord = [0,1]
        shock_pos = 1
        t_end = 1
        boundary = "wrap"
        initial_left = np.array([1,0,0,0,.1,0,0,0])
        initial_right = np.array([1,0,0,0,.1,0,0,0])
        misc = {'A':.9, 'ampl':np.sqrt(2)}

    # [Toro, 1999, p.225]
    elif "toro" in config:
        axis_coord = [0,1]
        boundary = "edge"
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
        axis_coord = [0,1]
        shock_pos = .5
        t_end = 2
        boundary = "wrap"

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
            initial_left = np.array([.5323,0,1.206,0,.3,0,0,0])
            initial_right = np.array([.138,1.206,1.206,0,.029,0,0,0])
            misc = {'bottom_left':np.array([1.5,0,0,0,1.5,0,0,0]), 'bottom_right':np.array([.5323,1.206,0,0,.3,0,0,0])}

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
        axis_coord = [0,1]
        shock_pos = .5
        t_end = .2
        boundary = "edge"
        initial_left = np.array([1,0,0,0,1,0,0,0])
        initial_right = np.array([.125,0,0,0,.1,0,0,0])
        misc = None

    return {
        'axis_coord':axis_coord,
        'shock_pos':shock_pos,
        't_end':t_end,
        'boundary':boundary.lower(),
        'misc':misc,
        'initial_left':initial_left,
        'initial_right':initial_right,
        'ds':{ax: np.abs(np.diff(axis_coord))/cells[ax] for ax in range(len(cells))},
    }
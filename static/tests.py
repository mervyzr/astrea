import numpy as np

##############################################################################
# Initial conditions for test configs
##############################################################################

# Primitive variables [rho, vx, vy, vz, P, Bx, By, Bz]
def generate_test_conditions(config_variables):
    config, cells, gamma = config_variables['config'], config_variables['cells'], config_variables['gamma']
    match = lambda match_type, substrings: match_type(substring in config for substring in substrings)

    ##############################################
    # Smooth
    ##############################################
    if config.startswith("sin"):
        axis_coord = [0,1]
        shock_pos = 1
        t_end = 1
        boundary = "wrap"
        init_cond = np.array([0,1,1,1,1,0,0,0])
        ambient = np.array([0,1,1,1,1,0,0,0])
        test_specifics = {'freq':2, 'ampl':.1, 'y_offset':2}

    elif config.startswith('gauss'):
        axis_coord = [-1,1]
        shock_pos = 1
        t_end = 2
        boundary = "wrap"
        init_cond = np.array([0,1,1,1,1,0,0,0])
        ambient = np.array([0,1,1,1,1,0,0,0])
        test_specifics = {'peak_pos':0, 'ampl':.75, 'fwhm':.08, 'y_offset':1}

    # [Roy et al., 2004]
    elif match(any, ["manufacture", "euler"]):
        axis_coord = [-.5,.5]
        shock_pos = 0
        t_end = 2
        boundary = "wrap"
        init_cond = np.array([1,.1,.2,.3,1,0,0,0])
        ambient = np.array([1,.1,.2,.3,1,0,0,0])
        test_specifics = {'freq':2*np.pi}

    # [Tóth, 2000]
    elif match(any, ["circular", "polarised", "alfven"]) or config == "cpaw":
        alpha = np.pi/6
        axis_coord = [0,1/np.cos(alpha)], [0,1/np.sin(alpha)]
        shock_pos = 0
        t_end = 2
        boundary = "wrap"
        init_cond = np.array([1,0,0,0,.1,0,0,0])
        ambient = np.array([1,0,0,0,.1,0,0,0])
        test_specifics = {'alpha':alpha, 'ampl':.1, 'wave':'moving'}

    elif config.startswith('sq'):
        axis_coord = [-1,1]
        shock_pos = 1/3
        t_end = .05
        boundary = "wrap"
        init_cond = np.array([1,1,0,0,1,0,0,0])
        ambient = np.array([.01,1,0,0,1,0,0,0])
        test_specifics = None

    ##############################################
    # Shocktubes
    ##############################################
    # [Sod, 1978]
    elif "sod" in config:
        axis_coord = [0,1]
        shock_pos = .5
        t_end = .2
        boundary = "edge"
        init_cond = np.array([1,0,0,0,1,0,0,0])
        ambient = np.array([.125,0,0,0,.1,0,0,0])
        test_specifics = None

    elif "slow" in config:
        axis_coord = [0,1]
        shock_pos = .5
        t_end = .08
        boundary = "edge"
        init_cond = np.array([5.6698,-1.5336,0,0,100,0,0,0])
        ambient = np.array([1,-10.5636,0,0,1,0,0,0])
        test_specifics = None

    # [Shu & Osher, 1989]
    elif match(any, ["shu", "osher"]) or config == "so":
        axis_coord = [-1,1]
        shock_pos = -.8
        t_end = .47
        boundary = "edge"
        init_cond = np.array([3.857143,2.629369,0,0,10.3333,0,0,0])
        ambient = np.array([0,0,0,0,1,0,0,0])
        test_specifics = {'freq':5, 'ampl':.2, 'y_offset':1}

    # [Ryu & Jones, 1995]
    elif match(any, ["ryu", "jones"]) or config == "rj":
        axis_coord = [-.5,.5]
        shock_pos = 0
        t_end = .15
        boundary = "edge"
        init_cond = np.array([1.08,1.2,.01,.5,.95,1/np.sqrt(np.pi),1.8/np.sqrt(np.pi),1/np.sqrt(np.pi)])
        ambient = np.array([1,0,0,0,1,1/np.sqrt(np.pi),2/np.sqrt(np.pi),1/np.sqrt(np.pi)])
        test_specifics = None

    # [Brio & Wu, 1988]
    elif match(any, ["brio", "wu"]) or config == "bw":
        axis_coord = [-.5,.5]
        shock_pos = 0
        t_end = .1
        boundary = "edge"
        init_cond = np.array([1,0,0,0,1,.75,1,0])
        ambient = np.array([.125,0,0,0,.1,.75,-1,0])
        test_specifics = None

    # [Toro, 1999, p.225]
    elif "toro" in config:
        axis_coord = [0,1]
        boundary = "edge"
        test_specifics = None

        # Double rarefaction wave
        if "2" in config:
            shock_pos = .5
            t_end = .14
            init_cond = np.array([1,-2,0,0,.4,0,0,0])
            ambient = np.array([1,2,0,0,.4,0,0,0])

        elif "3" in config:
            shock_pos = .5
            t_end = .012
            init_cond = np.array([1,0,0,0,1000,0,0,0])
            ambient = np.array([1,0,0,0,.01,0,0,0])

        elif "4" in config:
            shock_pos = .3
            t_end = .05
            init_cond = np.array([5.99924,19.5975,0,0,460.894,0,0,0])
            ambient = np.array([5.99242,-6.19633,0,0,46.095,0,0,0])

        elif "5" in config:
            shock_pos = .8
            t_end = .012
            init_cond = np.array([1,-19.59745,0,0,1000,0,0,0])
            ambient = np.array([1,-19.59745,0,0,.01,0,0,0])

        else:
            shock_pos = .3
            t_end = .2
            init_cond = np.array([1,.75,0,0,1,0,0,0])
            ambient = np.array([.125,0,0,0,.1,0,0,0])

    ##############################################
    # Blastwaves
    ##############################################
    # [Sedov, 1959]
    elif "sedov" in config:
        axis_coord = [-10,10]
        shock_pos = .5
        t_end = 2
        boundary = "wrap"
        init_cond = np.array([1,0,0,0,100,0,0,0])
        ambient = np.array([1,0,0,0,1e-12,0,0,0])
        test_specifics = None

    # [Felker & Stone, 2018]
    elif match(all, ["mhd", "blast"]):
        axis_coord = [-.5,.5]
        shock_pos = .1
        t_end = .2
        boundary = "wrap"
        init_cond = np.array([1,0,0,0,10,0,0,0])
        ambient = np.array([1,0,0,0,.1,0,0,0])
        test_specifics = {'ampl':np.sqrt(2*np.pi)}

    elif "noh" in config:
        axis_coord = [0,1]
        shock_pos = .1
        t_end = 1
        boundary = "edge"
        init_cond = np.array([1,0,0,0,1e-6,0,0,0])
        ambient = np.array([16,0,0,0,16/3,0,0,0])
        test_specifics = None

    ##############################################
    # Vortices
    ##############################################
    # [Gresho & Chan, 1990]
    elif "gresho" in config:
        axis_coord = [-.5,.5]
        shock_pos = 0
        t_end = 1
        boundary = "wrap"
        init_cond = np.array([1,0,0,0,0,0,0,0])
        ambient = np.array([1,0,0,0,0,0,0,0])
        test_specifics = {'mach':.1}

    # [Balsara, 2004; Li, 2010]
    elif match(all, ["mhd", "vortex"]):
        boundary = "wrap"
        t_end = 20

        if len(cells) < 3:
            axis_coord = [-10,10]
            shock_pos = 10
            init_cond = np.array([1,0,0,0,1,0,0,0])
            ambient = np.array([1,0,0,0,1,0,0,0])
            test_specifics = {'kappa':5, 'mu':5}
        else:
            axis_coord = [-5,5]
            shock_pos = 5
            init_cond = np.array([1,0,0,2,1,0,0,0])
            ambient = np.array([1,0,0,2,1,0,0,0])
            test_specifics = {'kappa':1/np.sqrt(2*np.pi), 'mu':1/np.sqrt(2*np.pi), 'q':1}

    # [Orszag & Tang, 1998; Stone et al., 2008; Pang & Wu, 2025]
    elif match(any, ["orszag", "tang"]) or config == "ot":
        axis_coord = [0,1]
        shock_pos = 0
        t_end = 1
        boundary = "wrap"
        init_cond = np.array([gamma**2,0,0,0,gamma,0,0,0])
        ambient = np.array([gamma**2,0,0,0,gamma,0,0,0])
        test_specifics = {'norm_factor':2*np.pi, 'ampl':1, 'eps':.2}

    # [Pang & Wu, 2025]
    elif match(any, ["ivc", "isentropic"]):
        axis_coord = [0,10]
        shock_pos = 5
        t_end = 10
        boundary = "wrap"
        init_cond = np.array([1,0,0,0,1,0,0,0])
        ambient = np.array([1,0,0,0,1,0,0,0])
        test_specifics = {'vortex_str':1, 'freq':2}

    ##############################################
    # Instabilities
    ##############################################
    elif match(any, ["kelvin", "helmholtz", "khi"]):
        axis_coord = [-1,1]
        shock_pos = .5
        t_end = 2
        boundary = "wrap"
        init_cond = np.array([2,.5,0,0,2.5,0,0,0])
        ambient = np.array([1,-.5,0,0,2.5,0,0,0])
        test_specifics = {'perturb':True, 'ampl':.5, 'freq':4, 'Bx':np.sqrt(np.pi)/2}

    elif match(any, ["rayleigh", "taylor", "rti"]):
        axis_coord = [-.5,.5]
        shock_pos = 0
        t_end = 10
        boundary = "edge"
        init_cond = np.array([2,.0,0,0,2.5,0,0,0])
        ambient = np.array([1,0,0,0,2.5,0,0,0])
        test_specifics = {'perturb':True, 'ampl':.05, 'grav_acc':.1, 'Bx':.05*np.sqrt(np.pi)}

    ##############################################
    # Turbulent/random noise
    ##############################################
    # Uniform field with turbulent driving motions [Federrath et al., 2010; Brucy et al., 2024]
    elif match(any, ["turb", "blank"]):
        axis_coord = [-.5,.5]
        shock_pos = 0
        t_end = .5
        boundary = "wrap"
        init_cond = np.array([gamma**2,0,0,0,gamma,0,0,0])
        ambient = np.array([gamma**2,0,0,0,gamma,0,0,0])
        test_specifics = {
            'zeta':.5, 'mach':5.5, 'f_rms':50, 'k_range':[1,3],
            'magnetic':True, 'mag_ampl':.1/np.sqrt(4*np.pi), 
            'perturb_ampl':.1
            }

    ##############################################
    # Non-linear MHD
    ##############################################
    # [Balsara & Spicer, 1999; Pang & Wu, 2025]
    elif match(any, ["rotor", "blob"]):
        axis_coord = [-.5,.5]
        boundary = "wrap"
        t_end = .5
        shock_pos = .1

        # Keplerian rotating blob
        if "blob" in config:
            init_cond = np.array([gamma**2,0,0,0,gamma,0,0,0])
            ambient = np.array([gamma**2/10,0,0,0,gamma/10,0,0,0])
            test_specifics = {
                'omega': 2.5, 
                'B_ampl': .1, 
                'rotation_axis': [0,0]  # theta, phi (deg)
            }
        else:
            init_cond = np.array([10,0,0,0,.5,2.5,0,0])
            ambient = np.array([1,0,0,0,.5,2.5,0,0])
            test_specifics = {'omega':1, 'ring_width':.015}

    # [Gardiner & Stone, 2005]
    elif match(any, ["current", "sheet"]):
        axis_coord = [-1,1]
        shock_pos = .5
        t_end = 10
        boundary = "wrap"
        init_cond = np.array([1,0,0,0,.1,0,1,0])
        ambient = np.array([1,0,0,0,.1,0,1,0])
        test_specifics = {'ampl':.1}

    ##############################################
    # Astrophysical
    ##############################################
    # [Dai & Woodward, 1998; Pang & Wu, 2025]
    elif "cloud" in config:
        axis_coord = [0,1]
        shock_pos = .6
        t_end = 1
        boundary = "edge"
        init_cond = np.array([3.86859,0,0,0,167.345,0,2.1826182,-2.1826182])
        ambient = np.array([1,-11.2536,0,0,1,0,.56418958,.56418958])
        test_specifics = {'cloud_mass':10}

    # [Wu & Shu, 2018]
    elif "jet" in config:
        axis_coord = [-.5,.5]
        shock_pos = .05
        t_end = .01
        boundary = "edge"
        init_cond = np.array([gamma*.1,0,0,0,1,0,np.sqrt(20),0])
        ambient = np.array([gamma*.1,0,0,0,1,0,np.sqrt(20),0])
        test_specifics = {'perturb':False, 'velocity':800}

    # [Machida et al., 1999]
    elif "torus" in config:
        axis_coord = [-7,7]
        shock_pos = 0
        t_end = 5
        boundary = "edge"
        init_cond = np.array([1e-3,0,0,0,0,0,0,0])
        ambient = np.array([1e-5,0,0,0,0,0,0,0])
        test_specifics = {'K':.05, 'B_phi':1, 'GM':1, 'L':1, 'r0':1, 'beta0':1}

    # [Markert et al., 2022]
    elif match(any, ["supernova", "tycho"]) or config == "sn":
        # tau0 is the constant in Skumanich's law for stellar rotation
        # (assumed Sun's age and rotational velocity)
        rotation = False
        tau0 = .1708284534  # [(pc/yr)⋅yr^0.51]
        age = 1e7

        mode = 'full'  # quadrant/octant or full-sphere mode
        shock_pos = 0
        t_end = 490
        init_cond = np.array([2.4539e-3,0,0,0,2.1309e-13,0,0,0])
        ambient = np.array([2.4539e-3,0,0,0,2.1309e-13,0,0,0])
        test_specifics = {'E':5.2516e-5, 'M':1.4, 't0':10, 'rotation':rotation, 'tau0':tau0, 'age':age, 'mode':mode}

        if mode.lower().startswith(('o','q')):
            axis_coord = [0,5]
            boundary = "reflect"
        else:
            axis_coord = [-5,5]
            boundary = "edge"

    ##############################################
    # 2D Riemann
    ##############################################
    # [Yee & Sjögreen, 2005]
    elif match(any, ["yee", "sjögreen", "sjoegreen"]) or config == "ys":
        axis_coord = [-1,1]
        shock_pos = 0
        t_end = 1
        boundary = "wrap"
        init_cond = np.array([1.0304,1.5308618,-1.0146545,-.09860248,2.48552123,.3501,.5078,.1576])
        ambient = np.array([.9308,1.56392351,-.49774388,0.06177482,2.27014061,.3501,.983,.305])
        test_specifics = {
            'bottom_left':np.array([1,1.75,-1,0,2.4322841,.5642,.5078,.2539]), 
            'bottom_right':np.array([1.8887,.12357706,-.92243342,.03880976,6.20869473,.5642,.983,.4915])
            }

    # [Lax & Liu, 1998]
    elif match(any, ["lax", "liu", "ll"]):
        axis_coord = [0,1]
        shock_pos = .5
        t_end = 2
        boundary = "wrap"

        if "ll" in config:
            index = int(config.replace(' ','').split('ll')[-1])
        else:
            index = int(config.replace(' ','').split('liu')[-1])

        if index in [1, 2]:
            init_cond = np.array([.5197,-.7259,0,0,.4,0,0,0])
            ambient = np.array([1,0,0,0,1,0,0,0])
            if index == 1:
                test_specifics = {'bottom_left':np.array([.1072,-.7259,-1.4045,0,.0439,0,0,0]), 'bottom_right':np.array([.2579,0,-1.4045,0,.15,0,0,0])}
            else:
                test_specifics = {'bottom_left':np.array([1,-.7259,-.7259,0,1,0,0,0]), 'bottom_right':np.array([.5197,0,-.7259,0,.4,0,0,0])}

        elif index == 3:
            init_cond = np.array([.5323,0,1.206,0,.3,0,0,0])
            ambient = np.array([.138,1.206,1.206,0,.029,0,0,0])
            test_specifics = {'bottom_left':np.array([1.5,0,0,0,1.5,0,0,0]), 'bottom_right':np.array([.5323,1.206,0,0,.3,0,0,0])}

        elif index == 4:
            init_cond = np.array([.5065,.8939,0,0,.35,0,0,0])
            ambient = np.array([1.1,0,0,0,1.1,0,0,0])
            test_specifics = {'bottom_left':np.array([1.1,.8939,.8939,0,1.1,0,0,0]), 'bottom_right':np.array([.5065,0,.8939,0,.35,0,0,0])}

        elif index in [5, 6]:
            coeff = -1**index
            init_cond = np.array([2,coeff*.75,.5,0,1,0,0,0])
            ambient = np.array([1,coeff*.75,-.5,0,1,0,0,0])
            test_specifics = {'bottom_left':np.array([1,-coeff*.75,.5,0,1,0,0,0]), 'bottom_right':np.array([3,-coeff*.75,-.5,0,1,0,0,0])}

        elif index == 7:
            init_cond = np.array([.5197,-.6259,.1,0,.4,0,0,0])
            ambient = np.array([1,.1,.1,0,1,0,0,0])
            test_specifics = {'bottom_left':np.array([.8,.1,.1,0,.4,0,0,0]), 'bottom_right':np.array([.5197,.1,-.6259,0,.4,0,0,0])}

        elif index == 8:
            init_cond = np.array([1,-.6259,.1,0,1,0,0,0])
            ambient = np.array([.5197,.1,.1,0,.4,0,0,0])
            test_specifics = {'bottom_left':np.array([.8,.1,.1,0,1,0,0,0]), 'bottom_right':np.array([1,.1,-.6259,0,1,0,0,0])}

        elif index == 9:
            init_cond = np.array([2,0,-.3,0,1,0,0,0])
            ambient = np.array([1,0,.3,0,1,0,0,0])
            test_specifics = {'bottom_left':np.array([1.039,0,-.8133,0,.4,0,0,0]), 'bottom_right':np.array([.5197,0,-.4259,0,.4,0,0,0])}

        elif index == 10:
            init_cond = np.array([.5,0,.6076,0,1,0,0,0])
            ambient = np.array([1,0,.4297,0,1,0,0,0])
            test_specifics = {'bottom_left':np.array([.2281,0,-.6076,0,.3333,0,0,0]), 'bottom_right':np.array([.4562,0,-.4297,0,.3333,0,0,0])}

        elif index == 11:
            init_cond = np.array([.5313,.8276,0,0,.4,0,0,0])
            ambient = np.array([1,.1,0,0,1,0,0,0])
            test_specifics = {'bottom_left':np.array([.8,.1,0,0,.4,0,0,0]), 'bottom_right':np.array([.5313,.1,.7276,0,.4,0,0,0])}

        elif index == 12:
            init_cond = np.array([1,.7276,0,0,1,0,0,0])
            ambient = np.array([.5313,0,0,0,.4,0,0,0])
            test_specifics = {'bottom_left':np.array([.8,0,0,0,1,0,0,0]), 'bottom_right':np.array([1,0,.7276,0,1,0,0,0])}

        elif index == 13:
            init_cond = np.array([2,.3,0,0,1,0,0,0])
            ambient = np.array([1,0,-.3,0,1,0,0,0])
            test_specifics = {'bottom_left':np.array([1.0625,0,.8145,0,.4,0,0,0]), 'bottom_right':np.array([.5313,0,.4276,0,.4,0,0,0])}

        elif index == 14:
            init_cond = np.array([1,0,-1.2172,0,8,0,0,0])
            ambient = np.array([2,0,-.5606,0,8,0,0,0])
            test_specifics = {'bottom_left':np.array([.4736,0,1.2172,0,2.6667,0,0,0]), 'bottom_right':np.array([.9474,0,1.1606,0,2.6667,0,0,0])}

        elif index == 15:
            init_cond = np.array([.5197,-.6259,-.3,0,.4,0,0,0])
            ambient = np.array([1,.1,-.3,0,1,0,0,0])
            test_specifics = {'bottom_left':np.array([.8,.1,-.3,0,.4,0,0,0]), 'bottom_right':np.array([.5313,.1,.4276,0,.4,0,0,0])}

        elif index == 16:
            init_cond = np.array([1.0222,-.6179,.1,0,1,0,0,0])
            ambient = np.array([.5313,.1,.1,0,.4,0,0,0])
            test_specifics = {'bottom_left':np.array([.8,.1,.1,0,1,0,0,0]), 'bottom_right':np.array([1,.1,.8276,0,1,0,0,0])}

        elif index in [17, 18, 19]:
            if index == 17:
                v1, v4 = -.4, -1.1259
            elif index == 18:
                v1, v4 = 1, .2741
            else:
                v1, v4 = .3, -.4259
            init_cond = np.array([2,0,-.3,0,1,0,0,0])
            ambient = np.array([1,0,v1,0,1,0,0,0])
            test_specifics = {'bottom_left':np.array([1.0625,0,.2145,0,.4,0,0,0]), 'bottom_right':np.array([.5197,0,v4,0,.4,0,0,0])}

    else:
        axis_coord = [0,1]
        shock_pos = .5
        t_end = .2
        boundary = "edge"
        init_cond = np.array([1,0,0,0,1,0,0,0])
        ambient = np.array([.125,0,0,0,.1,0,0,0])
        test_specifics = None

    if all(isinstance(coord, list) for coord in axis_coord):
        coordinates = {ax: coord for ax, coord in enumerate(axis_coord)}
    else:
        coordinates = {ax: axis_coord for ax in range(len(cells))}
    ds = {ax: np.abs(np.diff(coordinates[ax]))/cells[ax] for ax in range(len(cells))}

    return {
        'shock_pos':shock_pos,
        't_end':t_end,
        'boundary':boundary.lower(),
        'init_cond':init_cond,
        'ambient':ambient,
        'test_specifics':test_specifics,
        'coordinates':coordinates,
        'ds':ds,
    }
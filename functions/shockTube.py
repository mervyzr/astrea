import numpy as np

from functions import fv

##############################################################################

class shockTube:
    def __init__(self, _configVariables, _testVariables):
        for k, v in _configVariables.items():
            setattr(self, k, v)
        for k, v in _testVariables.items():
            setattr(self, k, v)


    # Initialise the discrete solution array/shocktube with initial conditions and primitive variables w
    # Returns the solution array in conserved variables q
    def initialise(self):
        arr = np.zeros((self.cells, len(self.initialRight)), dtype=self.precision)
        arr[:] = self.initialRight

        midpoint = (self.startPos+self.endPos)/2
        if self.config == "sedov" or self.config.startswith('sq'):
            half_width = int(self.cells/2 * ((self.shockPos-midpoint)/(self.endPos-midpoint)))
            left_edge, right_edge = int(self.cells/2-half_width), int(self.cells/2+half_width)
            arr[left_edge:right_edge] = self.initialLeft
        else:
            split_point = int(self.cells * ((self.shockPos-self.startPos)/(self.endPos-self.startPos)))
            arr[:split_point] = self.initialLeft

        if "shu" in self.config or "osher" in self.config:
            xi = np.linspace(self.shockPos, self.endPos, self.cells-split_point)
            arr[split_point:,0] = 1 + (.2 * np.sin(self.freq*np.pi*xi))
        else:
            xi = np.linspace(self.startPos, self.endPos, self.cells)
            if self.config == "sin":
                arr[:,0] = 1 + (.1 * np.sin(self.freq*np.pi*xi))
            elif self.config == "sinc":
                arr[:,0] = np.sinc(xi * self.freq/np.pi) + 1
            else:
                arr[:,0] = 1e-3 + (1-1e-3)*np.exp(-(xi-midpoint)**2/.01)
        
        self.domain = fv.pointConvertPrimitive(arr, self.gamma)
        return self.domain
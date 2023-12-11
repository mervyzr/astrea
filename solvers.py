import numpy as np

import functions as fn
import slope_limiters as limiters


# Piecewise constant Lax-Friedrichs solver (1st-order stable)
class LFSolver:
    def __init__(self, g):
        self.gamma = g
        self.eigmax = 0

    # Jacobian matrix using primitive variables
    def makeJacobian(self, tube):
        rho, vx, pressure = tube[:,0], tube[:,1], tube[:,4]
        arr = np.zeros((len(tube), 5, 5))  # create empty square arrays for each cell
        i,j = np.diag_indices(5)
        arr[:,i,j], arr[:,0,1], arr[:,1,4], arr[:,4,1] = vx[:,None], rho, 1/rho, self.gamma*pressure  # replace matrix with values
        return arr

    # Make f_i based on initial conditions and primitive variables
    def makeFlux(self, tube):
        rhos, vecs, pressures = tube[:,0], tube[:,1:4], tube[:,4]
        return np.c_[rhos*vecs[:,0], rhos*(vecs[:,0]**2) + pressures, rhos*vecs[:,0]*vecs[:,1], rhos*vecs[:,0]*vecs[:,2],\
                     vecs[:,0] * ((.5*rhos*np.linalg.norm(vecs, axis=1)**2) + ((self.gamma*pressures)/(self.gamma-1)))]

    # Calculate Riemann flux
    def calculateRiemannFlux(self, qLs, qRs):
        wLs, wRs = fn.convertConservative(qLs, self.gamma), fn.convertConservative(qRs, self.gamma)
        fLs, fRs = self.makeFlux(wLs), self.makeFlux(wRs)

        AL, AR = np.nan_to_num(self.makeJacobian(wLs), copy=False), np.nan_to_num(self.makeJacobian(wRs), copy=False)
        eigvalL, eigvalR = np.linalg.eigvals(AL), np.linalg.eigvals(AR)
        eigval = max(np.max(abs(eigvalL)), np.max(abs(eigvalR)))

        if eigval > self.eigmax:
            self.eigmax = eigval  # Compute the maximum wave speed; the maximum wave speed is the max eigenvalue between all cells i and i-1

        return .5 * ((fLs+fRs) - (eigval*(qRs-qLs)))
    

# Piecewise linear Godunov solver (2nd-order stable)
class GSolver:
    def __init__(self, g):
        self.gamma = g
        self.eigmax = 0

    # Jacobian matrix using primitive variables
    def makeJacobian(self, tube):
        rho, vx, pressure = tube[:,0], tube[:,1], tube[:,4]
        arr = np.zeros((len(tube), 5, 5))  # create empty square arrays for each cell
        i,j = np.diag_indices(5)
        arr[:,i,j], arr[:,0,1], arr[:,1,4], arr[:,4,1] = vx[:,None], rho, 1/rho, self.gamma*pressure  # replace matrix with values
        return arr

    # Make f_i based on initial conditions and primitive variables
    def makeFlux(self, tube):
        rhos, vecs, pressures = tube[:,0], tube[:,1:4], tube[:,4]
        return np.c_[rhos*vecs[:,0], rhos*(vecs[:,0]**2) + pressures, rhos*vecs[:,0]*vecs[:,1], rhos*vecs[:,0]*vecs[:,2],\
                     vecs[:,0] * ((.5*rhos*np.linalg.norm(vecs, axis=1)**2) + ((self.gamma*pressures)/(self.gamma-1)))]

    # Calculate Riemann flux
    def calculateRiemannFlux(self, qLs, qRs):
        gradients = limiters.minmod(qLs, qRs)

        interfaceLefts, interfaceRights = qRs[:-1]+gradients, qLs[1:]-gradients

        wLs, wRs = fn.convertConservative(qLs, self.gamma), fn.convertConservative(qRs, self.gamma)
        fLs, fRs = self.makeFlux(wLs), self.makeFlux(wRs)

        AL, AR = np.nan_to_num(self.makeJacobian(wLs), copy=False), np.nan_to_num(self.makeJacobian(wRs), copy=False)
        eigvalL, eigvalR = np.linalg.eigvals(AL), np.linalg.eigvals(AR)
        eigval = max(np.max(abs(eigvalL)), np.max(abs(eigvalR)))

        if eigval > self.eigmax:
            self.eigmax = eigval  # Compute the maximum wave speed; the maximum wave speed is the max eigenvalue between all cells i and i-1

        return .5 * ((fLs+fRs) - (eigval*(qRs-qLs)))    

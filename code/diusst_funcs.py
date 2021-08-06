"""
diusst_funcs.py
Custom functions needed to run diusst
"""

import numpy as np
from scipy.optimize import fsolve

###############################################################################

def find_eps(eps,N,dz0,z_f):
    return -dz0/z_f*(1-eps**(N-1)) - eps +1

def make_mesh(dz0,N,z_f=10, verbose=False):
    z = np.zeros(N+2)
    z[0] = dz0

    eps = fsolve(find_eps,1.5,args=(N+1,dz0,z_f))[0]

    for i in range(2,N+2):
        z[i] = - dz0*(1-eps**(i-1))/(1-eps)

    if verbose:
        print('Stretch factor for dz='+str(dz0)+' and '+str(N)+' dynamic grid points: '+str(round(eps,3)))

    return z, eps

#################################################################################

def dndz(z,dz0=0.05,eps=1):
    return 1/(np.log(eps)*(dz0/(1-eps)+z))

def dndz2(z,dz0=0.05,eps=1):
    return -1/(np.log(eps)*(dz0/(1-eps)+z)**2)

#################################################################################

def laplace_central(a,deriv1,deriv2):
    # a must be N+2 array, with ghost row at top and bottom
    b = np.zeros(len(a))

    b[1] = (a[2]-a[1]) * (deriv1[1]**2 + 0.5*deriv2[1])
    b[2:-1] = (a[3:] - 2*a[2:-1] + a[1:-2]) * deriv1[2:-1]**2 + (a[3:]-a[1:-2])*0.5*deriv2[2:-1]

    return b

def grad_central(a, deriv1):
    # a must be N+2 array, with ghost row at top and bottom
    b = np.zeros(len(a))

    b[1:-1] = (a[:-2]-a[2:]) * deriv1[1:-1] / 2

    return b

def grad_backward(a, deriv1):
    # a must be N+2 array, with ghost row at top and bottom
    b = np.zeros(len(a))

    b[1:-1] = (a[1:-1]-a[:-2]) * deriv1[1:-1]

    return b


def laplace_central_uni(a,dz):
    # a must be N+2 array, with ghost row at top and bottom
    b = np.zeros(len(a))

    b[1] = (a[2]-a[1]) / dz**2
    b[2:-1] = (a[3:] - 2*a[2:-1] + a[1:-2]) / dz**2

    return b

def grad_backward_uni(a,dz):
    # a must be N+2 array, with ghost row at top and bottom
    b = np.zeros(len(a))

    b[1:-1] = -(a[1:-1]-a[:-2]) / dz

    return b

#################################################################################

# Refraction angle based on Snell's law
def snell(theta, n1=1., n2=1.34):
    theta = theta % (2*np.pi)
    theta = np.where((theta > (np.pi/2)) & (theta < (3*np.pi/2)), np.pi/2, theta)

    return np.abs(np.arcsin(n1/n2*np.sin(theta)))

#################################################################################

# compute saturation specific humidity from temperature
def s_sat(T, rho_a, R_v):
    return 100 * 6.112 * np.exp( 17.67*(T-273.15)/(T-273.15+243.5) ) / (rho_a * R_v * T)

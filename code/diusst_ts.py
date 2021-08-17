'''
1D Diurnal SST model for timeseries
last edited: May 18, 2021
Author: Reyk
'''

# import required packages
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import inv
from mscfunctions import *

# Diurnal SST model
def diusst_ts(
    Ta_data,                        # air temperature data array
    sw_data,                        # incoming shortwave radiation data array
    time,                           # array of time points (in s)
    t_sim=None,                     # duration of simulation (in s)
    dt=None,                        # time step (in s)
    z_col=10,                       # depth of modeled water column (in m)
    dz=0.02,                        # depth resolution (if streched grid, input np.array of grid spacing instead of float) (in m)
    sa_data=None,                   # (optional) air specific humidity data array (in kg/kg)
    u_data=None,                    # (optional) wind speed data array (in m/s)
    wind=0,                         # constant wind speed (if u_data is None) (in m/s)
    T_0=300,                        # foundation temperature (in K)
    cloud=0,                        # constant cloud cover fraction
    mixing=0.04/3/60,               # linear mixing coefficient mu (in 1/s)
    diffu=1.,                       # relative strength of diffusion (molecular and turbulent)
    sens=1.,                        # relative strength of sensible heat flux
    latent=1.,                      # relative strength of latent heat flux
    attenu=0.8,                     # light attenuation coefficient (in 1/m)
    opac=1.,                        # opacity of the vertically integrated atmosphere to longwave radiation
    k_mol = 1e-7,                   # molecular diffusivity of sea water (in m^2/s)
    k_eddy = 1e-4,                  # eddy diffusivity coefficient (in m^2/s)
    n_w = 1.34,                     # refractive index of sea water
    n_a = 1.0,                      # refractive index of air
    rho_w = 1027,                   # density of sea water (in kg/m^3)
    rho_a = 1.1,                    # density of air (in kg/m^3)
    humid = 10*1e-3,                # constant specific humidity at reference height above sea level (if sa_data is None) (in kg/kg)
    C_s = 1.3e-3,                   # turbulent exchange coefficient for sensible heat
    C_l = 1.5e-3,                   # turbulent exchange coefficient for latent heat
    L_evap = 2.5e6,                 # latent heat of evaporization (J/kg) Source: Klose p.151
    c_p = 3850,                     # specific heat of sea water at const pressure (in J/K/kg)
    c_p_a = 1005,                   # specific heat of air at const pressure (in J/K/kg)
    sb_const = 5.67e-8,             # Stefan Boltzmann constant (in W/m^2/K^4)
    R_v = 461.51,                   # gas constant of water vapor, J/K/kg):
    wind_dep = 'quadratic',         # set the dependence of eddy diffusivity on wind speed ('quadratic' or 'cubic')
    mixing_type = 1,                # parametrization of convective mixing. 1=constant, 2=propto distance from foundation, 3=propto skin effect
    skin = False,
    output = 1):                    # model output. 1=full temperature array, 2=difference skin-foundation, 3=difference column max-foundation

    # Define depth mesh
    if type(dz) == float:
        N_z = int(z_col / dz) + 2
        z = - np.arange(0, z_col + 2*dz, dz) + dz/2

    else:
        N_z = dz.shape[0] + 1
        z = np.ones(N_z)*dz[0]/2
        for i in range(1,N_z):
            z[i] = z[i-1] - dz[i-1]

    # number of layers (excluding the dummy layers at air interface and foundation)
    N = N_z - 2

    # Define time array
    #N_t = int(t_sim / dt)
    #t = np.arange(0,t_sim, dt)
    t = time
    N_t = len(t)
    dt = t[1:]-t[:-1]

    # Atmospheric input data
    if sa_data is None:
        s_a = np.ones(N_t) * humid
    else:
        s_a = sa_data

    if u_data is None:
        u = np.ones(N_t) * wind
    else:
        u = u_data

    # Initialize temperature array and heat flux arrays
    init = np.ones(N_z)*T_0
    T = np.array([init[1:-1]])
    Qs, Ql, Rlw, Rsw = [], [], [], []

    # Eddy diffusivity
    if wind_dep == 'quadratic':
        kappa  = diffu*(k_mol + k_eddy*u**2)
    elif wind_dep == 'cubic':
        kappa  = diffu*(k_mol + k_eddy*u**3)

    # Time integration
    Tn = T
    for n in range(1, N_t):

        #solar angle
        solangle = 2*np.pi*n*dt[n-1]/(60*60*24)+np.pi

        if mixing_type == 1:
            mix = np.ones(N) * mixing
        elif mixing_type == 2:
            mix = mixing / np.abs(z[1:-1]+z_col)
        elif mixing_type == 3:
            skin = np.amax(Tn) - Tn[0]
            mix = np.ones(N) * mixing * skin

        elif mixing_type == 4:
            mix = mixing / np.abs(z[1:-1]+z_col)
            angle = solangle-np.pi/4
            if np.cos(angle) > 0:
                kappa[n] += k_eddy*np.cos(angle)**3

        alpha, betap, betam, gamma = np.zeros(N), np.zeros(N), np.zeros(N), np.ones(N)

        if type(dz) == float:
            deltap = 2*dz**2
            deltam = deltap
            gamma = gamma * dt[n-1]/(dz*rho_w*c_p)

        for i in range(N):
            if type(dz) != float:
                deltap = dz[i]**2 + dz[i]*dz[i+1]
                deltam = dz[i+1]**2 + dz[i]*dz[i+1]
                gamma[i] = dt[n-1] / (dz[i] * rho_w * c_p)

            alpha[i] = 1 + mix[i]*dt[n-1] + 2*kappa[n] * dt[n-1] / dz**2 #(1/deltam + 1/deltap)
            betap[i] = - kappa[n] * dt[n-1] / dz**2
            betam[i] = - kappa[n] * dt[n-1] / dz**2

        if skin:
            alpha[0] = 1 + mix[0]*dt[n-1] + 2*diffu*k_mol * dt[n-1] / dz**2

        ent1, ent2, ent3 = alpha, betap[:-1], betam[1:]
        row1, row2, row3 = np.arange(N), np.arange(N-1), np.arange(1,N)

        row = np.concatenate((row1, row2, row3))
        col = np.concatenate((row1, row3, row2))
        ent = np.concatenate((ent1, ent2, ent3))

        A = sparse.csc_matrix((ent, (row,col)), shape=(N,N))
        A_inv = inv(A).toarray()

        # air-sea fluxes into ocean (positive downwards)
        R_sw = (1-cloud) * sw_data[n] * np.exp(attenu/np.cos(snell(solangle))*z)
        R_lw = sb_const * (opac*(Ta_data[n])**4 - (T[-1,0])**4)
        Q_s  = sens * rho_a * c_p_a * C_s * max(0.5, u[n]) * (Ta_data[n] - T[-1,0])
        Q_l  = latent * rho_a * L_evap * C_l * max(0.5, u[n]) * (s_a[n] - s_sat(T[-1,0], rho_a, R_v))

        # solution vector
        b     = gamma * (R_sw[:-2] - R_sw[1:-1]) + mix*dt[n-1]*T_0
        b[0] += gamma[0] * (R_lw + Q_l + Q_s) - betam[0] * T[-1,0]
        b[-1] = - betap[-1] * T_0 + mix[-1]*dt[n-1]*T_0

        # solution at time step n
        Tn = np.dot(A_inv, (T[n-1] + b))

        # store result in output arrays
        T = np.concatenate((T, np.array([Tn])), axis=0)
        Qs.append(Q_s)
        Ql.append(Q_l)
        Rlw.append(R_lw)
        Rsw.append(R_sw[0])

    if output==1:
        return [T, z, t, [Qs, Ql, Rlw, Rsw]]

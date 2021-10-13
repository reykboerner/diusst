"""
DiuSST model
Featuring the BASE, LIN, EXP, and STAB versions
Documentation:
Author: Reyk
Date: 13 Oct 2021
"""

# import required packages
import numpy as np
from diusst_funcs import make_mesh, snell, s_sat, laplace_central, grad_central, grad_backward, dndz, dndz2

def diusst(
    # Atmospheric input data
    time_data,                  # array of time points (in s)
    wind_data,                  # wind speed data array (in m/s)
    swave_data,                 # incoming shortwave radiation data array (in W/m2)
    airtemp_data,               # air temperature data array (in K)
    humid_data,                 # air specific humidity data array (in kg/kg)
    # Model parameters
    T_f=300,                    # foundation temperature (in K)
    kappa = 1e-4,               # eddy diffusivity coefficient (in m^2/s)
    mu=1,                       # linear mixing coefficient μ (in 1/s)
    alpha=1,                    # light attenuation coefficient (in 1/m)
    lambd=3,                    # trapping depth (in m)
    sigma=0.8,                  # surface suppressivity
    z_ref=1,                    # reference depth for STAB model
    # Model options
    diffu_type='BASE',          # Type of vertical diffusivity profile
    init = None,                # initial sea temperature of domain. if None, then T_f is used as the domain temperature (in K)
    output = 'basic',           # What information the function outputs
    wind_max = 10,              # Cut-off wind speed
    wind_exp = 2,               # exponent of the wind speed in turbulent diffusivity
    opac=1,                     # opacity of the vertically integrated atmosphere to longwave radiation
    # Model domain
    z_f=10,                     # Foundation depth (in m)
    dz=0.05,                    # depth resolution (if streched grid, this is the thickness of the top layer) (in m)
    ngrid = None,               # number of vertical grid points (excluding foundation and air dummy). If None, then a uniform mesh is generated with grid spacing dz
    # Constants
    k_mol = 1e-7,               # molecular diffusivity of sea water (in m^2/s)
    cp_w = 3850,                # specific heat of sea water at const pressure (in J/K/kg)
    cp_a = 1005,                # specific heat of air at const pressure (in J/K/kg)
    rho_w = 1027,               # density of sea water (in kg/m^3)
    rho_a = 1.1,                # density of air (in kg/m^3)
    n_w = 1.34,                 # refractive index of sea water
    n_a = 1.0,                  # refractive index of air
    C_s = 1.3e-3,               # turbulent exchange coefficient for sensible heat
    C_l = 1.5e-3,               # turbulent exchange coefficient for latent heat
    L_evap = 2.5e6,             # latent heat of evaporization (J/kg) Source: Klose p.151
    sb_const = 5.67e-8,         # Stefan Boltzmann constant (in W/m^2/K^4)
    gas_const = 461.51):        # gas constant of water vapor, J/K/kg):

    # Define vertical grid
    N_z = ngrid + 2
    z, stretch = make_mesh(dz,ngrid,z_f=z_f)
    zidx_ref = np.where(z>-z_ref)[0][-1] # used only for STAB model

    # Stretched grid derivatives
    dv1 = dndz(z, dz0=dz,eps=stretch)
    dv2 = dndz2(z, dz0=dz,eps=stretch)

    # Define time stepping
    N_t = len(time_data)
    dt = time_data[1:]-time_data[:-1]

    # Initialize temperature, heat flux, and diffusivity arrays
    T      = np.zeros((N_t,N_z))    # Water temperature
    diffu  = np.zeros((N_t,N_z))    # Eddy diffusivity
    ddiffu = np.zeros((N_t,N_z))    # Spatial derivative of eddy diffusivity
    R_sw   = np.zeros((N_t,N_z))    # Shortwave heat flux
    Q_s, Q_l, R_lw = [], [], []     # Sensible, latent, and longwave heat fluxes

    # Initial condition
    if init is None:
        T[0] = np.ones(N_z) * T_f
    else:
        T[0] = np.ones(N_z) * init

    # Boundary condition
    T[:,-1] = np.ones(N_t) * T_f

    # Mixing coefficient
    mix = np.zeros(N_z)
    mix[1:-1] = mu / np.abs(z[1:-1]-z[-1]) * np.abs(z[2:]-z[1:-1])

    # Shortwave heat flux
    solar_angle = 2*np.pi*time_data/86400 + np.pi
    for i in range(N_z):
        R_sw[:,i] = swave_data * np.exp(alpha/np.cos(snell(solar_angle))*z[i])

    # Generate wind speed array capped at wind_max and taken to the power of wind_exp
    wind_factor = np.minimum(wind_data,wind_max)**wind_exp

    # Eddy diffusivity
    if diffu_type == 'BASE':
        for i in range(N_z):
            diffu[:,i] = k_mol + wind_factor * kappa

    elif diffu_type == 'LIN':
        for i in range(N_z):
            diffu[:,i] = k_mol + wind_factor * kappa * np.abs(z[i]/z_f)
            ddiffu[:,i] = - wind_factor * kappa / z_f

    elif diffu_type == 'EXP':
        for i in range(N_z):
            diffu[:,i] = k_mol + wind_factor * kappa * (1-sigma*np.exp(z[i]/lambd))/(1-k_0*np.exp(-z_f/lambd))
            ddiffu[:,i] = - wind_factor * kappa / lambd * sigma*np.exp(z[i]/lambd) /(1-k_0*np.exp(-z_f/lambd))

    # Time integration
    for n in range(1, N_t):

        # Compute surface fluxes
        Rlw = sb_const * (opac*(airtemp_data[n-1])**4 - (T[n-1,1])**4)
        Qs  = rho_a * cp_a * C_s * max(0.5, wind_data[n-1]) * (airtemp_data[n-1] - T[n-1,1])
        Ql  = rho_a * L_evap * C_l * max(0.5, wind_data[n-1]) * (humid_data[n-1] - s_sat(T[n-1,1], rho_a, gas_const))

        # Total heat flux
        Q = R_sw[n-1]
        Q[0] += (Rlw + Qs + Ql)

        # STAB model
        if diffu_type == 'STAB':
            S = min( max((T[n-1,1]-T[n-1,zidx_ref]), 0) , 1)
            diffu[n-1] = k_mol + wind_factor[n-1] * kappa * (1-S*sigma*np.exp(z/lambd))/(1-S*sigma*np.exp(-z_f/lambd))
            ddiffu[n-1] = - wind_factor[n-1] * kappa * S*sigma/lambd * np.exp(z/lambd) /(1-S*sigma*np.exp(-z_f/lambd))

        # Euler integration
        T[n] = T[n-1] + dt[n-1] * (
                diffu[n-1]*laplace_central(T[n-1],dv1,dv2)
                + ddiffu[n-1]*grad_central(T[n-1],dv1)
                - mix*(T[n-1]-T_f)
                + grad_backward(Q,dv1)/(rho_w*cp_w))

        # Write heat fluxes
        if output == 'detailed':
            R_lw.append(Rlw)
            Q_s.append(Qs)
            Q_l.append(Ql)

    if output == 'temp':
        return T[:,1:]
    elif output == 'basic':
        return [T[:,1:], z[1:], time_data]
    elif output == 'detailed':
        return [T[:,1:], z[1:], time_data, [Q_s, Q_l, R_lw, R_sw[:,1]]]

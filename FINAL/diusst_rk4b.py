"""
Diurnal SST model for Bayesian analysis
Code boiled down to necessities in order to minimize computiation time
cloud = 0
wind_dep = quadratic
diffu_type = 'linear'
skin = False
diffu = 1
sens = 1
latent = 1
rad = 1
surflux = 1
"""

# import required packages
import numpy as np
from diusst_funcs import make_mesh, snell, s_sat, laplace_central, grad_central, grad_backward, dndz, dndz2

def diusst_rk4b(
    times,                          # array of time points (in s)
    Ta_data,                        # air temperature data array
    sw_data,                        # incoming shortwave radiation data array
    u_data=None,                    # (optional) wind speed data array (in m/s)
    sa_data=None,                   # (optional) air specific humidity data array (in kg/kg)
    T_f=300,                        # foundation temperature (in K)
    z_f=10,                         # depth of modeled water column (in m)
    dz=0.05,                        # depth resolution (if streched grid, this is the thickness of the top layer) (in m)
    attenu=1,                       # light attenuation coefficient (in 1/m)
    mu=1,                           # linear mixing coefficient μ (in 1/s)
    k_eddy = 1e-4,                  # eddy diffusivity coefficient (in m^2/s)
    # constants
    k_mol = 1e-7,                   # molecular diffusivity of sea water (in m^2/s)
    n_w = 1.34,                     # refractive index of sea water
    n_a = 1.0,                      # refractive index of air
    rho_w = 1027,                   # density of sea water (in kg/m^3)
    rho_a = 1.1,                    # density of air (in kg/m^3)
    C_s = 1.3e-3,                   # turbulent exchange coefficient for sensible heat
    C_l = 1.5e-3,                   # turbulent exchange coefficient for latent heat
    L_evap = 2.5e6,                 # latent heat of evaporization (J/kg) Source: Klose p.151
    c_p = 3850,                     # specific heat of sea water at const pressure (in J/K/kg)
    c_p_a = 1005,                   # specific heat of air at const pressure (in J/K/kg)
    sb_const = 5.67e-8,             # Stefan Boltzmann constant (in W/m^2/K^4)
    R_v = 461.51,                   # gas constant of water vapor, J/K/kg):
    # optional
    t_sim=None,                     # duration of simulation (in s)
    dt=None,                        # time step (in s)
    init = None,                    # initial sea temperature of domain. if None, then T_f is used as the domain temperature (in K)
    wind=0,                         # constant wind speed (if u_data is None) (in m/s)
    humid = 10*1e-3,                # constant specific humidity at reference height above sea level (if sa_data is None) (in kg/kg)
    # switches
    maxwind = 10,
    ngrid = None,                   # number of vertical grid points (excluding foundation and air dummy). If None, then a uniform mesh is generated with grid spacing dz
    opac=1,):                       # opacity of the vertically integrated atmosphere to longwave radiation

    # Define depth mesh
    N_z = ngrid + 2
    z, stretch = make_mesh(dz,ngrid,z_f=z_f)

    # Define time array
    N_t = len(times)
    dt = times[1:]-times[:-1]

    # Atmospheric input data
    s_a = sa_data
    u = u_data

    # Initialize temperature array and heat flux arrays
    T = np.zeros((N_t, N_z))

    # initial condition
    T[0] = np.ones(N_z)*T_f

    # boundary condition
    T[:,-1] = np.ones(N_t)*T_f

    # solar angle
    solangle = 2*np.pi*times/86400 + np.pi

    # initialize diffusivity, mixing coefficient, heat fluxes
    kappa = np.zeros((N_t,N_z))
    mix = np.zeros(N_z)
    R_sw = np.zeros((N_t,N_z))
    Qs, Ql, Rlw = [], [], []

    # mixing coefficient
    mix[1:-1] = mu * np.abs(z[2:]-z[1:-1]) / np.abs(z[1:-1]-z[-1])

    for i in range(N_z):
        # diffusivity
        kappa[:,i] = k_mol - k_eddy * z[i]/z_f *(np.minimum(maxwind,u))**2
        #solar radiation
        R_sw[:,i] = sw_data * np.exp(attenu/np.cos(snell(solangle))*z[i])

    # stretched grid prerequisites
    dv1 = dndz(z, dz0=dz,eps=stretch)
    dv2 = dndz2(z, dz0=dz,eps=stretch)

    # Time integration
    for n in range(1, N_t):

        # surface fluxes
        R_lw = sb_const * (opac*(Ta_data[n-1])**4 - (T[n-1,0])**4)
        Q_s  = rho_a * c_p_a * C_s * max(0.5, u[n-1]) * (Ta_data[n-1] - T[n-1,0])
        Q_l  = rho_a * L_evap * C_l * max(0.5, u[n-1]) * (s_a[n-1] - s_sat(T[n-1,0], rho_a, R_v))

        # Total heat flux
        Q = R_sw[n-1]
        #Q[0] = rad*R_sw[n-1,1] + surflux*(R_lw + Q_s + Q_l)
        Q[0] += (R_lw + Q_s + Q_l)

        # RK4 integration
        Tt = T[n-1]
        k1 = dt[n-1] * (kappa[n-1]*laplace_central(Tt,dv1,dv2) - k_eddy/z_f *(min(u[n-1],maxwind))**2*grad_central(Tt,dv1) - mix*(Tt-T_f) + grad_backward(Q,dv1)/(rho_w*c_p))
        Tt = T[n-1] + k1/2
        k2 = dt[n-1] * (kappa[n-1]*laplace_central(Tt,dv1,dv2) - k_eddy/z_f *(min(u[n-1],maxwind))**2*grad_central(Tt,dv1) - mix*(Tt-T_f) + grad_backward(Q,dv1)/(rho_w*c_p))
        Tt = T[n-1] + k2/2
        k3 = dt[n-1] * (kappa[n-1]*laplace_central(Tt,dv1,dv2) - k_eddy/z_f *(min(u[n-1],maxwind))**2*grad_central(Tt,dv1) - mix*(Tt-T_f) + grad_backward(Q,dv1)/(rho_w*c_p))
        Tt = T[n-1] + k3
        k4 = dt[n-1] * (kappa[n-1]*laplace_central(Tt,dv1,dv2) - k_eddy/z_f *(min(u[n-1],maxwind))**2*grad_central(Tt,dv1) - mix*(Tt-T_f) + grad_backward(Q,dv1)/(rho_w*c_p))

        T[n] = T[n-1] + (k1 + 2*k2 + 2*k3 +k4) / 6

    return [T[:,1:], z[1:], times]
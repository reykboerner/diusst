import numpy as np

def slab(times, Ta_data, u, swrad, s_a, d=1, sink=100, Tf=300, xi1=0.001, xi2=5e-9):

    T = np.ones(len(times))*Tf
    dt = times[1:]-times[:-1]

    n_w = 1.34                     # refractive index of sea water
    n_a = 1.0                     # refractive index of air
    rho_w = 1027                   # density of sea water (in kg/m^3)
    rho_a = 1.1                    # density of air (in kg/m^3)
    C_s = 1.3e-3                   # turbulent exchange coefficient for sensible heat
    C_l = 1.5e-3                   # turbulent exchange coefficient for latent heat
    L_evap = 2.5e6                 # latent heat of evaporization (J/kg) Source: Klose p.151
    c_p = 3850                     # specific heat of sea water at const pressure (in J/K/kg)
    c_p_a = 1005                   # specific heat of air at const pressure (in J/K/kg)
    sb_const = 5.67e-8             # Stefan Boltzmann constant (in W/m^2/K^4)
    R_v = 461.51
    opac=1

    corr = 0

    for n in range(1,len(times)):
        #print(u[n-1])
        corr = corr + (T[n-1] - Tf)*dt[n-1]

        R_lw = sb_const * (opac*(Ta_data[n-1])**4 - (T[n-1])**4)
        Q_s  = rho_a * c_p_a * C_s * max(0.5, u[n-1]) * (Ta_data[n-1] - T[n-1])
        Q_l  = rho_a * L_evap * C_l * max(0.5, u[n-1]) * (s_a[n-1] - s_sat(T[n-1], rho_a, R_v))

        dT = dt[n-1]/(rho_w*c_p*d)*(swrad[n]+R_lw+Q_s+Q_l-sink) - xi1 * (T[n-1]-Tf) - xi2 *corr


        T[n] = T[n-1]+dT

    return T

__author__ = "Reyk Boerner"

'''
Implementation of the Zeng & Beljaars 2005 model.
Author: Reyk Börner (reyk.boerner@reading.ac.uk)
Date: 01 March 2023
'''

import numpy as np
from tqdm import tqdm

class ZengBeljaars:
    """
    Python implementation of the Zeng & Beljaars 2005 diurnal SST scheme.
    Includes a cool skin layer scheme based on Fairall et al. 1996.
    See documentation: https://github.com/reykboerner/diusst

    References:
    [1] Zeng and Beljaars (2005), https://doi.org/10.1029/2005GL023030
    [2] Fairall et al. (1996), https://dx.doi.org/10.1029/95JC03190
    [3] COARE 3.6 algorithm, https://github.com/NOAA-PSL/COARE-algorithm/tree/master/Python/COARE3.6,
        specifically 'coare36vn_zrf_et.py', release v1.1, lines 568-573
    """

    def __init__(self, takaya10=True, reflect=False,
        T_f = 300,                      # foundation temperature (K)
        d = 3,                          # warm layer depth (m)
        nu = 0.3,                       # profile shape parameter
        k = 0.4,                        # von Karman constant
        g = 9.81,                       # gravity (m/s²)
        k_w = 1e-7,                     # molecular diffusion coeff. (m²/s)
        a_w = 2.9e-4,                   # thermal expansion coeff. of seawater (1/K)
        nu_w = 9e-7,                    # kinematic viscosity of seawater (m²/s)
        rho_w = 1027,                   # density of seawater (kg/m³)        
        rho_a = 1.1,                    # density of air (kg/m³)        
        cp_w = 3850,                    # specific heat capacity of seawater (J/(kg K))
        cp_a = 1005,                    # specific heat capacity of air (J/(kg K))
        n_w = 1.34,                     # refractive index of seawater
        n_a = 1.0,                      # refractive index of air
        C_s = 1.3e-3,                   # Stanton number
        C_l = 1.5e-3,                   # Dalton number
        L_evap = 2.5e6,                 # Latent heat of vaporization (J/kg)
        sb_const = 5.67e-8,             # Stefan-Boltzmann constant (W/(m K²)²)
        gas_const = 461.51,             # gas constant of water vapor (J/(kg K))
        rad_a = [0.28, 0.27, 0.45],     # coefficients of radiation absorption scheme
        rad_b = [71.5, 2.8, 0.07],      # exponents of radiation absorption scheme (1/m)
        C_drag = 1.3e-3,                # drag coefficient to compute wind stress
        z_wind = 10,                    # wind speed measurement height above sea level (m)
        z_rough = None):                # roughness height of sea surface (m)

        self.T_f = T_f
        self.d = d
        self.nu = nu
        self.k = k
        self.g = g
        self.k_w = k_w
        self.a_w = a_w
        self.nu_w = nu_w        
        self.rho_w = rho_w
        self.rho_a = rho_a
        self.cp_w = cp_w
        self.cp_a = cp_a
        self.n_w = n_w
        self.n_a = n_a
        self.C_s = C_s
        self.C_l = C_l
        self.L_evap = L_evap
        self.sb_const = sb_const
        self.gas_const = gas_const
        self.rad_a = rad_a
        self.rad_b = rad_b    
        self.takaya10 = takaya10        # Refinements of Takaya et al. (2010)
        self.reflect = reflect          # surface reflection of incident radiation
        self.C_drag = C_drag
        self.z_wind = z_wind
        self.z_rough = z_rough

    def friction_vel(self, u):
        """
        Calculates the friction velocity in the water, given wind speed u at height z above
        the surface.
        z0 is the roughness height of the surface (e.g., average wave height).
        Source: https://www.calculatoratoz.com/en/eniction-veloceny-for-known-wind-speed-at-height-above-surface-calculator/calc-23770
        and https://en.wikipedia.org/wiki/Shear_velocity
        and Takaya et al. (2010)
        """
        if self.z_rough is None:
            wind_stress = self.rho_a*self.C_drag*u**2
            return np.sqrt(wind_stress/self.rho_w)
        else:
            return self.k*u/np.log(self.z_wind/self.z_rough)*np.sqrt(self.rho_a/self.rho_w)
    
    def stability_function(self, x):
        """Eq. (9) of Zeng and Beljaars 2005 [1]"""
        if x >= 0:
            if self.takaya10:
                return 1 + (5*x + 4*x**2)/(1 + 3*x + 0.25*x**2)
            else:
                return 1 + 5*x
        else:
            return (1 - 16*x)**(-0.5)
    
    def sst_skin(self, data, tidx, Ts, delta):
        """Eq. (4) of Zeng and Beljaars 2005 [1]"""
        heatflux = (self.surface_flux(data, tidx, Ts) 
            + self.f_s(delta)*self.shortwave_net(data, tidx, self.reflect))
        prefactor = delta/(self.rho_w*self.cp_w*self.k_w)
        return prefactor*heatflux

    def surface_flux(self, data, tidx, Ts, sum=True):
        """Net surface heat flux (positive into the ocean)"""
        flux = {}
        flux["Rlw"] = self.sb_const*(data.T_a_rel[tidx]**4 - Ts**4)
        flux["Qs"] = self.rho_a*self.cp_a*self.C_s*data.u[tidx]*(data.T_a_rel[tidx] - Ts)
        flux["Ql"] = self.rho_a*self.L_evap*self.C_l*data.u[tidx]*(data.q_v[tidx] -
            self.get_sat_humid(Ts))
        
        if sum:
            return flux["Rlw"] + flux["Qs"] + flux["Ql"]
        else:
            return flux

    def shortwave_net(self, data, tidx, reflect):
        """Downward shortwave radiation penetrating the sea surface"""
        if reflect:
            print("ERROR: SW rad reflection at surface not implemented")
        else:
            return data.Q_sw[tidx]
    
    def get_sat_humid(self, T):
        """Calculates saturation specific humidity for given air temperature T, in kg/kg."""
        return 611.2*np.exp(17.67*(T - 273.15)/(T - 29.65))/(self.rho_a*self.gas_const*T)
    
    def f_s(self, delta):
        """Eq. (5) of Zeng and Beljaars 2005 [1]"""
        return 0.065 + 11*delta - 6.6e-5/delta*(1 - np.exp(-delta/(8e-4)))

    def skin_thickness(self, data, tidx, Ts, delta):
        """Eq. (6) of Zeng and Beljaars 2005 [1]
        with correction according to Fairall et al. 1996 eqs. (12) and (14) [2]
        and modification according to the COARE3.6 algorithm [3]"""
        heatflux = (self.surface_flux(data, tidx, Ts) 
            + self.f_s(delta)*self.shortwave_net(data, tidx, self.reflect))
        u_fric = self.friction_vel(data.u[tidx])
        factor = - (16*self.g*self.a_w*self.nu_w**3)/(u_fric**4*self.k_w**2*self.rho_w*self.cp_w)
        delta_new = 6*(1 + (max(0, factor)*heatflux)**(3/4))**(-1/3)*self.nu_w/u_fric
        return min(0.01, delta_new)

    def R(self, data, tidx):
        """Radiation absorption scheme (see Zeng and Beljaars 2005 [1], Soloviev 1982)"""
        R_s = self.shortwave_net(data, tidx, self.reflect)
        sum = 0
        for i in range(3):
            sum += self.rad_a[i]*np.exp(-self.d*self.rad_b[i])
        return R_s*sum

    def L_MO(self, data, tidx, Ts, dT):
        """Monin-Obukhov length, eq. (10) of Zeng and Beljaars 2005 [1]"""
        u_fric = self.friction_vel(data.u[tidx])
        return (self.rho_w*self.cp_w*u_fric**3)/(self.k*self.F_d(data, tidx, Ts, dT))
        
    def F_d(self, data, tidx, Ts, dT):
        """Buoyancy flux, eq. (12) of Zeng and Beljaars 2005 [1]"""
        if dT <= 0:
            heatflux = (self.surface_flux(data, tidx, Ts) 
                + self.shortwave_net(data, tidx, self.reflect)
                - self.R(data, tidx))
            return self.g*self.a_w*heatflux
        else:
            u_fric = self.friction_vel(data.u[tidx])
            factor = np.sqrt(self.nu*self.g*self.a_w/(5*self.d))*self.rho_w*self.cp_w
            return factor*u_fric**2*np.sqrt(dT)

    def dsst_dot(self, data, tidx, Ts, dT):
        """RHS of eq. (11) of Zeng and Beljaars 2005 [1]"""
        heatflux = (self.surface_flux(data, tidx, Ts) 
                + self.shortwave_net(data, tidx, self.reflect)
                - self.R(data, tidx))
        stab_func = self.stability_function(self.d/self.L_MO(data, tidx, Ts, dT))
        u_fric = self.friction_vel(data.u[tidx])
        term1 = heatflux/(self.d*self.rho_w*self.cp_w*self.nu/(self.nu + 1))        
        term2 = (self.nu + 1)*self.k*u_fric*dT/(self.d*stab_func)
        return term1 - term2
    
    def simulate(self, data, init, skin_iter=6, output="T"):
        """
        Evolves the sea skin and subskin temperature forward in time, given an atmospheric
        dataset 'data' and initial condition 'init'.

        Input
        ------
        * 'data': dataset of type AtmosData
        * 'init': array of initial conditions [x, y, delta], where
            - 'x' is the subskin temperature deviation (T_subskin - T_f) in K
            - 'y' is the cool skin temperature effect (T_skin - T_subskin) in K
            - 'delta' is the skin layer thickness in m
        
        Output
        ------
        [x, y, delta]: array of arrays
        
        The sea skin temperature is given by T_f + x + y.
        """
        N = len(data.time)
        dt = data.time[1:] - data.time[:-1]

        x = np.ones(N)*init[0]
        y = np.ones(N)*init[1]
        delta = np.ones(N)*init[2]
        flux = np.zeros((N,3))

        for i in tqdm(range(1,N)):
            # previous skin temperature
            Ts = self.T_f + x[i-1] + y[i-1]
            # compute f_s
            fs = self.f_s(delta[i-1])
            # compute skin effect
            y[i] = self.sst_skin(data, i-1, Ts, delta[i-1])
            # compute bulk warming
            x[i] = x[i-1] + dt[i-1]*self.dsst_dot(data, i-1, Ts, x[i-1])
            # compute delta
            _delta = delta[i-1]
            for k in range(skin_iter):
                _delta = self.skin_thickness(data, i-1, Ts, _delta)
            delta[i] = _delta

            if output == "all":
                flux[i] = np.fromiter(self.surface_flux(data, i, Ts, sum=False).values(),
                dtype=float)
            
        if output == "all":
            return x, y, delta, [flux[:,j] for j in range(3)]
        elif output == "T":
            return x, y
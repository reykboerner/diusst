__author__ = 'Reyk Boerner'

import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
from tqdm import tqdm

from utils import _grad_bckward, _grad_forward, _grad_central, _lapl_central

class Diusst:
    """
    DiuSST, a diurnal sea surface temperature model.
    Documentation: https://github.com/reykboerner/diusst#readme

    Attributes
    ----------

    diffu_profile: str
        Functional form of the vertical diffusivity profile.
        Options are 'CONST', 'LIN', 'EXP', 'S-LIN', 'S-EXP'.

    reflect: bool
        Switch on/off reflection of downward shortwave radiation at the sea surface.

    Methods
    ----------

    simulate(data, options)
        Run the DiuSST model forced by a given atmospheric data set.

    interpolate(data, options)
        Interpolate a given data set in time, with variable time step.
        Time steps are calculated for specified model parameters and wind conditions
        in order to satisfy the CFL condition at all times.

    get_mesh(args)
        Generate a stretched 1D mesh along the vertical coordinate for finite difference scheme.

    get_sat_humid(args)
        Calculate the saturation specific humidity for given air temperature.

    """

    def __init__(self, diffu_profile='LIN', reflect=True,
        T_f = 300,
        kappa = 1.42e-4,
        mu = 2.93e-3,
        alpha = 4.01,
        lambd = 3,
        sigma = 0.8,
        z_ref = 1,
        wind_max = 10,
        wind_exp = 2,
        CFL = 0.95,
        z_f = 10,
        dz0 = 0.10,
        ngrid = 40,
        k_mol = 1e-7,
        cp_w = 3850,
        cp_a = 1005,
        rho_w = 1027,
        rho_a = 1.1,
        n_w = 1.34,
        n_a = 1.0,
        C_s = 1.3e-3,
        C_l = 1.5e-3,
        L_evap = 2.5e6,
        sb_const = 5.67e-8,
        gas_const = 461.51):

        """
        Options
        ----------

        CFL: float
            Target CFL number when determining the variable time step during data interpolation.

        wind_max: float
            Cutoff maximum wind speed in the diffusion term, to limit computational cost.

        wind_exp: int
            Exponent in the relation between turbulent diffusivity and wind speed. Standard value is 2.

        Parameters
        ----------
        T_f:        (float) Foundation temperature (K).
        kappa:      (float) Eddy diffusivity (m^2/s).
        mu:         (float) Mixing coefficient (m/s).
        alpha:      (float) Attenuation coefficient (1/m).
        sigma:      (float) Surface suppressivity.
        lambd:      (float) Trapping depth (m).
        z_ref:      (float) Reference depth for stability function in S-model versions (m).

        Domain
        ----------
        z_f:        (float) Foundation depth (m).
        dz0:        (float) Vertical grid resolution at sea surface (m).
        ngrid:      (int)   Number of vertical grid points. If None, then uniform grid with spacing dz0 is used.

        Physical constants
        ----------
        k_mol       (float) Molecular diffusivity of sea water (m^2/s).
        cp_w        (float) Specific heat capacity at constant pressure of sea water (J/kg/K).
        cp_a        (float) Specific heat capacity at constant pressure of air (J/kg/K).
        rho_w       (float) Density of sea water (kg/m^3).
        rho_a       (float) Density of air (kg/m^3).
        n_w         (float) Refractive index of sea water.
        n_a         (float) Refractive index of air.
        C_s         (float) Stanton number.
        C_l         (float) Dalton number.
        L_evap      (float) Latent heat of vaporization (J/kg).
        sb_const    (float) Stefan-Boltzmann constant (W/m^2/K^4).
        gas_const   (float) Gas constant of water vapor (J/kg/K).
        """

        self.diffu_profile = diffu_profile
        self.surflux = surflux
        self.reflect = reflect
        self.T_f = T_f
        self.kappa = kappa
        self.mu = mu
        self.alpha = alpha
        self.sigma = sigma
        self.lambd = lambd
        self.z_ref = z_ref
        self.wind_max = wind_max
        self.wind_exp = wind_exp
        self.CFL = CFL
        self.z_f = z_f
        self.dz0 = dz0
        self.ngrid = ngrid
        self.k_mol = k_mol
        self.cp_w = cp_w
        self.cp_a = cp_a
        self.rho_w = rho_w
        self.rho_a = rho_a
        self.n_w = n_w
        self.n_a = n_a
        self.C_s = C_s
        self.C_l = C_l
        self.L_evap = L_evap
        self.sb_const = sb_const
        self.gas_const = gas_const

    ###############################################################################################
    ### Main function for running DiuSST ##########################################################
    ###############################################################################################

    def simulate(self, data, init=None, output=None, progress=True):
        """
        Run the DiuSST model under given atmospheric forcing.
        Numerical integration based on explicit Euler scheme

        Arguments
        ----------
        data: dict or Pandas DataFrame or xarray DataSet
            Atmosheric input data. Must contain time series of the following variables, labeled:
            'times':    Time since midnight in local time (s)
            'wind':     Wind speed (m/s)
            'swrad':    Incident solar irradiance (W/m^2)
            'atemp':    Air temperature (K)
            'humid':    Specific humidity (kg/kg)

        init: float
            Contant initial sea temperature in the domain. If None (default), then T_f is used.

        output: str
            What data to return.

        progress: bool
            Switch for tqdm progress bar.

        Returns
        ----------
        if 'output' is None:        [T, t, z]                       (list)
        if 'output' is 'detailed':  [T, t, z, [Qs, Ql, Rlw, Rsw]]   (list)

            T:      (2D array, time x depth) Sea temperature
            t:      (1D array) Time (s)
            z:      (1D array) Depth (m)
            Qs:     (1D array) Latent heat flux (1D array) (W/m^2)
            Ql:     (1D array) Sensible heat flux (1D array) (W/m^2)
            Rlw:    (1D array) Longwave radiative flux (W/m^2)
            Rsw:    (1D array) Shortwave radiative flux at air-sea interface (W/m^2)

        """

        # Extract data
        time_data = data['times'].to_numpy(np.float64)
        wind_data = data['wind'].to_numpy(np.float64)
        swrad_data = data['swrad'].to_numpy(np.float64)
        airtemp_data = data['atemp'].to_numpy(np.float64)
        humid_data = data['humid'].to_numpy(np.float64)

        # Define vertical grid
        if self.ngrid is None:
            self.ngrid = int(self.z_f/self.dz0)

        N_z = self.ngrid + 2
        z, stretch = Diusst.get_mesh(self)
        zidx_ref = np.where(z > -self.z_ref)[0][-1]

        # Stretched grid derivatives
        dv1 = Diusst._dndz(self)
        dv2 = Diusst._dndz2(self)

        # Time steps
        N_t = len(time_data)
        dt = time_data[1:]-time_data[:-1]

        # Initialize temperature and surface flux arrays
        T = np.zeros((N_t,N_z))
        Q_s, Q_l, R_lw = [], [], []

        # Initial condition
        if init is None:
            T[0] = np.ones(N_z) * self.T_f
        else:
            T[0] = np.ones(N_z) * init

        # Boundary condition at foundation depth
        T[:,-1] = np.ones(N_t) * self.T_f

        # Initialize mixing term
        mix = np.zeros(N_z)
        mix[1:-1] = self.mu / np.abs(z[1:-1]-z[-1]) * np.abs(z[2:]-z[1:-1])

        # Angle of the sun in the sky (with respect to surface normal)
        solar_angle = 2*np.pi/86400 * time_data + np.pi

        # Shortwave heat flux penetrating the sea surface
        R_sw = np.zeros((N_t,N_z))
        for i in range(N_z):
            R_sw[:,i] = Diusst._transmitted(self, swrad_data, solar_angle) * np.exp(self.alpha / np.cos( Diusst._refract_angle(self, solar_angle) ) * z[i])

        # Wind dependence of diffusion term
        wind_factor = np.minimum(wind_data, self.wind_max) ** self.wind_exp

        # Diffusion term (state-independent diffusivity profiles)
        diffu  = np.zeros((N_t,N_z))    # Eddy diffusivity
        ddiffu = np.zeros((N_t,N_z))    # Spatial derivative of eddy diffusivity

        if self.diffu_profile == 'CONST':
            for i in range(N_z):
                diffu[:,i] = self.k_mol + wind_factor * self.kappa

        elif self.diffu_profile == 'LIN':
            for i in range(N_z):
                diffu[:,i] = self.k_mol + wind_factor * self.kappa * (1 + self.sigma * (np.abs(z[i]/self.z_f) - 1))
                ddiffu[:,i] = - wind_factor * self.kappa * self.sigma / self.z_f

        elif self.diffu_profile == 'EXP':
            for i in range(N_z):
                diffu[:,i] = self.k_mol + wind_factor * self.kappa * (1 - self.sigma * np.exp(z[i]/self.lambd))/(1 - self.sigma*np.exp(-self.z_f/self.lambd))
                ddiffu[:,i] = - wind_factor * self.kappa / self.lambd * self.sigma * np.exp(z[i]/self.lambd) /(1 - self.sigma*np.exp(-self.z_f/self.lambd))

        ### Time integration ###
        if progress:
            itsteps = tqdm(range(1, N_t))
        else:
            itsteps = range(1, N_t)

        for n in itsteps:

            # Diffusion term (state-dependent diffusivity profiles)
            if self.diffu_profile == 'S-LIN':
                S = min( max((T[n-1,1]-T[n-1,zidx_ref]), 0) , 1)
                diffu[n-1] = self.k_mol + wind_factor[n-1] * self.kappa * (1 + S*self.sigma*(np.abs(z[i]/self.z_f)-1))
                ddiffu[n-1] = - wind_factor[n-1] * self.kappa * S * self.sigma / self.z_f

            elif self.diffu_profile == 'S-EXP':
                S = min( max((T[n-1,1]-T[n-1,zidx_ref]), 0) , 1)
                diffu[n-1] = self.k_mol + wind_factor[n-1] * self.kappa * (1-S*self.sigma*np.exp(z/self.lambd))/(1-S*self.sigma*np.exp(-self.z_f/self.lambd))
                ddiffu[n-1] = - wind_factor[n-1] * self.kappa * S*self.sigma/self.lambd * np.exp(z/self.lambd) /(1-S*self.sigma*np.exp(-self.z_f/self.lambd))

            # Compute surface fluxes
            Rlw = self.sb_const * (airtemp_data[n-1]**4 - T[n-1,1]**4)
            Qs  = self.rho_a * self.cp_a * self.C_s * wind_data[n-1] * (airtemp_data[n-1] - T[n-1,1])
            Ql  = self.rho_a * self.L_evap * self.C_l * wind_data[n-1] * (humid_data[n-1] - Diusst.get_sat_humid(self, T[n-1,1]))

            # Total heat flux
            Q = R_sw[n-1]
            Q[1] += Rlw + Qs + Ql

            # Euler step
            T[n] = T[n-1] + dt[n-1] * (
                diffu[n-1] * _lapl_central(T[n-1], dv1, dv2)
                + ddiffu[n-1] * _grad_central(T[n-1], dv1)
                - mix * (T[n-1] - self.T_f)
                + _grad_forward(Q, dv1) / (self.rho_w * self.cp_w)
                )

            # Write heat fluxes
            if output == 'detailed':
                R_lw.append(Rlw); Q_s.append(Qs); Q_l.append(Ql)

        if output is None:
            return [T[:,1:], time_data, z[1:]]

        elif output == 'detailed':
            return [T[:,1:], time_data, z[1:], [Q_s, Q_l, R_lw, R_sw[:,1]]]


    ###############################################################################################
    ### Variable time step interpolation algorithm ################################################
    ###############################################################################################

    def interpolate(self, data, save=None, verbose=True):
        """
        Variable time step interpolation of input data.
        Sets the time step such that the CFL number stays at the given constant value.

        Arguments
        ----------
        data: dict or Pandas DataFrame or xarray DataSet
            Atmosheric input data. Must contain time series of the following variables, labeled:
            'times':    Time since midnight in local time (s)
            'wind':     Wind speed (m/s)
            'swrad':    Incident solar irradiance (W/m^2)
            'atemp':    Air temperature (K)
            'humid':    Specific humidity (kg/kg)

        save: str
            Save the interpolated data set under the file name inserted for save.
            If None, then no file is saved.

        verbose: bool
            Control verbosity of output.

        Returns
        ----------
        [df_final, dt_list, idx_list]: list
            df_final:   (Pandas DataFrame) Interpolated data set.
            dt_list:    (list) List of computed time step durations.
            idx_list:   (list) List of indexes in interpolated data set corresponding to the original data points.

        """

        # Generate stretched mesh
        z = Diusst.get_mesh(self)[0]
        dz = (z[1:]-z[:-1])[:-1]

        # Extract data
        times = data['times'].to_numpy(np.float64)
        sst = data['sst'].to_numpy(np.float64)
        sst_err = data['sst_err'].to_numpy(np.float64)
        ftemp = data['ftemp'].to_numpy(np.float64)
        wind = data['wind'].to_numpy(np.float64)
        atemp = data['atemp'].to_numpy(np.float64)
        swrad = data['swrad'].to_numpy(np.float64)
        humid = data['humid'].to_numpy(np.float64)

        # Initialize arrays to store interpolated data
        series = np.stack((sst, sst_err, ftemp, wind, atemp, swrad, humid))
        times_concat = np.zeros(1)
        series_concat = np.zeros((7,1))
        dt_list = []
        idx_list = [0]

        # Interpolate from each data point to the next
        for i in range(1,len(times)):

            # Get wind speed
            u = min(self.wind_max, max(wind[i],wind[i-1]))

            # Calculate CFL/dt at all grid points
            if self.diffu_profile == 'CONST' or self.diffu_profile == 'S-EXP' or self.diffu_profile == 'S-LIN':
                c_array = 2 * (self.k_mol + self.kappa*u**self.wind_exp) / dz**2

            elif self.diffu_profile == 'LIN':
                c_array = 2 * (self.k_mol + self.kappa*u**self.wind_exp * (1 + self.sigma*(np.abs(z[1:-1]/self.z_f)-1))) / dz**2

            elif self.diffu_profile == 'EXP':
                c_array = 2 * (self.k_mol + self.kappa*u**self.wind_exp * (1-self.sigma*np.exp(z[1:-1]/self.lambd))/(1-self.sigma*np.exp(-self.z_f/self.lambd)) ) / dz**2

            else:
                raise AttributeError("diffu_profile not specified!")

            # Get maximum CFL and compute time step dt
            C = np.amax(c_array)
            dt = min(10, self.CFL/C)

            # Interpolate
            times_new = np.arange(times[i-1], times[i], dt)
            f = interp1d(times[i-1:i+1], series[:,i-1:i+1], fill_value="extrapolate")
            series_new = f(times_new)

            # Store interpolated data in array
            times_concat = np.concatenate((times_concat,times_new))
            series_concat = np.column_stack((series_concat,series_new))
            dt_list.append(dt)
            idx_list.append(len(times_concat)-1)

        # Create interpolated dataset
        final = np.row_stack((times_concat,series_concat)).transpose()
        df_final = pd.DataFrame(final[1:,:], columns=['times','sst','sst_err','ftemp','wind','atemp','swrad','humid'])

        # Option to save as CSV
        if save is not None:
            df_final.to_csv(save+'_data.csv')

        # Verbose output
        if verbose:
            print('Variable time-step interpolation at CFL = {}:'.format(self.CFL))
            print('---> Interpolated dataset has '+str(len(df_final))+' time steps with average length '+str(round(np.mean(dt_list),3))+' s.')
            print('---> Constant dt interpolation would require dt = '+str(round(np.amin(dt_list),3))+' s --> '+str(int((times[-1]-times[0])/np.amin(dt_list)))+' steps.')
            print('---> Computation time will be reduced by '+str(round((1-len(df_final)/int((times[-1]-times[0])/np.amin(dt_list)))*100,3))+' %.')

        return df_final, dt_list, idx_list[:-1]


    ###############################################################################################
    ### User functions ############################################################################
    ###############################################################################################

    def get_mesh(self):
        """
        Generates an array of depth coordinates for the stretched vertical grid,
        given dz0, ngrid, and z_f.
        """

        def _stretch_factor(epsilon):
            """
            Solves for the stretch factor of the vertical grid,
            given dz0, ngrid, and z_f.
            """
            return 1 - self.dz0 / self.z_f * (1 - epsilon**self.ngrid) - epsilon

        z = np.zeros(self.ngrid + 2)
        z[0] = self.dz0
        eps = fsolve(_stretch_factor, 1.5)[0]

        for i in range(2, len(z)):
            z[i] = - self.dz0 * (1 - eps**(i-1)) / (1 - eps)

        return z, eps

    def get_sat_humid(self, T):
        """
        Calculates saturation specific humidity for given air temperature T, in units kg/kg.
        """
        return 611.2 * np.exp(17.67 * (T - 273.15) / (T - 29.65) ) / (self.rho_a * self.gas_const * T)


    ###############################################################################################
    ### Internal functions ########################################################################
    ###############################################################################################

    def _refract_angle(self, angle):
        """
        Calculates the refraction of a light ray penetrating the air-sea interface, based on Snell's refraction law.

        Input:  incident solar angle (rad, with respect to surface normal)
        Output: refracted angle below sea surface (rad, with respect to surface normal)

        """

        # Set angle to pi/2 when sun is below horizon
        theta = angle % (2*np.pi)
        theta = np.where((theta > (np.pi/2)) & (theta < (3*np.pi/2)), np.pi/2, theta)

        return np.abs(np.arcsin(self.n_a / self.n_w * np.sin(theta)))

    def _transmitted(self, irradiance, angle):
        """
        Removes the fraction of downward shortwave radiation reflected at the sea surface.

        Input:
            irradiance (array): incident downward shortwave irradiance above sea surface
            angle (float): incident solar angle (rad, with respect to surface normal)
        """

        # Set angle to pi/2 when sun is below horizon
        theta = angle % (2*np.pi)
        theta = np.where((theta > (np.pi/2)) & (theta < (3*np.pi/2)), np.pi/2, theta)

        # Compute reflected fraction
        if self.reflect:
            sqrt = np.sqrt(1-(self.n_a/self.n_w*np.sin(theta))**2)
            perp = ((self.n_a*np.cos(theta) - self.n_w*sqrt)/(self.n_a*np.cos(theta) + self.n_w*sqrt))**2
            para = ((self.n_a*sqrt - self.n_w*np.cos(theta))/(self.n_a*sqrt + self.n_w*np.cos(theta)))**2
            reflected = (perp + para)/2
            return irradiance*(1-reflected)
        else:
            return irradiance

    def _dndz(self):
        """
        First stretched grid derivative (see eq. (5.26) of thesis).
        """

        z, eps = Diusst.get_mesh(self)
        return 1 / (np.log(eps) * (self.dz0 / (1-eps) + z))

    def _dndz2(self):
        """
        Second stretched grid derivative (see eq. (5.27) of thesis).
        """

        z, eps = Diusst.get_mesh(self)
        return - 1 / (np.log(eps) * (self.dz0 / (1-eps) + z)**2)

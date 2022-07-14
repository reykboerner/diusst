__author__ = 'Reyk Boerner'

'''
Source code for class Slab.
'''

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from tqdm import tqdm

class Slab:
    """
    Single-layer slab ocean model with proportional and integral corrector.
    Documentation: see https://arxiv.org/pdf/2205.07933.pdf

    Methods
    ----------

    simulate(data, options)
        Run the slab model forced by a given atmospheric data set.

    interpolate(data, options)
        Interpolate a given data set in time, with constant time step.

    get_sat_humid(args)
        Calculate the saturation specific humidity for given air temperature.

    """

    def __init__(self,
        d = 1,
        Q_sink = 100,
        xi_1 = 1e-3,
        xi_2 = 5e-9,
        T_f = 300,
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
        gas_const = 461.51,
        wind_max = 10):

        """
        Options
        ----------

        wind_max: float
            Cutoff maximum wind speed in the diffusion term, to limit computational cost.

        Parameters
        ----------
        d:          (float) Slab thickness (m).
        Q_sink:     (float) Constant heat sink (W/m^2).
        xi_1        (float) Proportional corrector strength.
        xi_2        (float) Integral corrector strength.
        T_f:        (float) Foundation temperature (K).

        Physical constants
        ----------
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

        self.d = d
        self.Q_sink = Q_sink
        self.xi_1 = xi_1
        self.xi_2 = xi_2
        self.T_f = T_f
        self.n_w = n_w
        self.n_a = n_a
        self.rho_w = rho_w
        self.rho_a = rho_a
        self.C_s = C_s
        self.C_l = C_l
        self.L_evap = L_evap
        self.cp_w = cp_w
        self.cp_a = cp_a
        self.sb_const = sb_const
        self.gas_const = gas_const
        self.wind_max = wind_max


    def simulate(self, data, output='T', progress=True):
        """
        Run the slab model under given atmospheric forcing.
        Numerical integration based on explicit Euler scheme.

        Arguments
        ----------
        data: dict or Pandas DataFrame or xarray DataSet
            Atmosheric input data. Must contain time series of the following variables, labeled:
            'time':         Time since midnight in local time (s)
            'wind':         Wind speed (m/s)
            'swrad':        Incident solar irradiance (W/m^2)
            'atemp_rel':    Air temperature relative to foundation temperature (K)
            'humid':        Specific humidity (kg/kg)

        output: str
            What data to return.

        progress: bool
            Switch for tqdm progress bar.

        Returns
        ----------
        if 'output' is 'T':         T                    (1D array)
        if 'output' is 'detailed':  [T, [Qs, Ql, Rlw]]   (list)

            T:      (1D array) Slab temperature as a function of time
            Qs:     (1D array) Latent heat flux (1D array) (W/m^2)
            Ql:     (1D array) Sensible heat flux (1D array) (W/m^2)
            Rlw:    (1D array) Longwave radiative flux (W/m^2)
        """

        # Extract data
        time_data = data['time'].to_numpy()
        wind_data = data['wind'].to_numpy()
        swrad_data = data['swrad'].to_numpy()
        airtemp_data = data['atemp_rel'].to_numpy()
        humid_data = data['humid'].to_numpy()

        T = np.ones(len(time_data)) * self.T_f
        dt = time_data[1:] - time_data[:-1]

        corr = 0
        Q_s, Q_l, R_lw = [], [], []

        if progress:
            itsteps = tqdm(range(1,len(time_data)))
        else:
            itsteps = range(1,len(time_data))

        for n in itsteps:

            corr += (T[n-1] - self.T_f) * dt[n-1]

            Rlw = self.sb_const * ((airtemp_data[n-1])**4 - (T[n-1])**4)
            Qs  = self.rho_a * self.cp_a * self.C_s * wind_data[n-1] * (airtemp_data[n-1] - T[n-1])
            Ql  = self.rho_a * self.L_evap * self.C_l * wind_data[n-1] * (humid_data[n-1] - Slab.get_sat_humid(self, T[n-1]))

            dT = (swrad_data[n-1] + Rlw + Qs + Ql - self.Q_sink) / (self.rho_w * self.cp_w * self.d) - self.xi_1 * (T[n-1]-self.T_f) - self.xi_2 * corr

            T[n] = T[n-1] + dt[n-1]*dT

            if output=='detailed':
                R_lw.append(Rlw); Q_s.append(Qs); Q_l.append(Ql)

        if output=='T':
            return T
        elif output=='detailed':
            return T, [np.array(Q_s), np.array(Q_l), np.array(R_lw)]

    def get_sat_humid(self, T):
        """
        Calculates saturation specific humidity for given air temperature T, in units kg/kg.
        (See eqs. (5.6) and (5.7) of thesis.)
        """
        return 611.2 * np.exp(17.67 * (T - 273.15) / (T - 29.65) ) / (self.rho_a * self.gas_const * T)

    def interpolate(self, data, dt=1, save=None, verbose=True):
        """
        Interpolates the input data with constant time step dt.

        Arguments
        ----------
        data: dict or Pandas DataFrame or xarray DataSet
            Atmosheric input data. Must contain time series of the following variables, labeled:
            'time':         Time since midnight in local time (s)
            'wind':         Wind speed (m/s)
            'swrad':        Incident solar irradiance (W/m^2)
            'atemp_rel':    Air temperature relative to foundation temperature (K)
            'humid':        Specific humidity (kg/kg)

        dt: float
            Time step between interpolated data points (in seconds).

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

        # Extract data from input file
        times = data['time'].to_numpy()
        sst = data['skinsst'].to_numpy()
        sst_err = data['dsst_err'].to_numpy()
        ftemp = data['ftemp'].to_numpy()
        wind = data['wind'].to_numpy()
        atemp = data['atemp_rel'].to_numpy()
        swrad = data['swrad'].to_numpy()
        humid = data['humid'].to_numpy()

        # Initialize arrays to store interpolated data
        series = np.stack((sst, sst_err, ftemp, wind, atemp, swrad, humid))
        times_concat = np.zeros(1)
        series_concat = np.zeros((7,1))
        dt_list = []
        idx_list = [0]

        # Interpolate from each data point to the next
        for i in range(1,len(times)):

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
        df_final = pd.DataFrame(final[1:,:], columns=['time','skinsst','dsst_err','ftemp','wind','atemp_rel','swrad','humid'])

        # Option to save as CSV
        if save is not None:
            df_final.to_csv(save+'_data.csv')

        # Verbose output
        if verbose:
            print('+++ Constant time-step interpolation +++')
            print('Interpolated dataset has '+str(len(df_final))+' time steps with length '+str(round(np.mean(dt_list),3))+' s.')
            print('++++++++++++++++++++++++++++++++++++++++')

        return df_final, dt_list, idx_list[:-1]

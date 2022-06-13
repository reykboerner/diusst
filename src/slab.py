__author__ = 'Reyk Boerner'

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from tqdm import tqdm

class Slab:
    """docstring for Slab."""

    def __init__(self,
        d = 1,
        Q_sink = 100,
        xi_1 = 1e-3,
        xi_2 = 5e-9,
        T_f = 300,
        n_w = 1.34,                     # refractive index of sea water
        n_a = 1.0,                      # refractive index of air
        rho_w = 1027,                   # density of sea water (in kg/m^3)
        rho_a = 1.1,                    # density of air (in kg/m^3)
        C_s = 1.3e-3,                   # turbulent exchange coefficient for sensible heat
        C_l = 1.5e-3,                   # turbulent exchange coefficient for latent heat
        L_evap = 2.5e6,                 # latent heat of evaporization (J/kg) Source: Klose p.151
        cp_w = 3850,                    # specific heat of sea water at const pressure (in J/K/kg)
        cp_a = 1005,                    # specific heat of air at const pressure (in J/K/kg)
        sb_const = 5.67e-8,             # Stefan Boltzmann constant (in W/m^2/K^4)
        gas_const = 461.51,
        opac=1,
        wind_max = 10
    ):
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
        self.opac = opac
        self.wind_max = wind_max


    def simulate(self, data, output='T', progress=True):

        # Extract data
        time_data = data['times'].to_numpy(np.float64)
        wind_data = data['wind'].to_numpy(np.float64)
        swrad_data = data['swrad'].to_numpy(np.float64)
        airtemp_data = data['atemp'].to_numpy(np.float64)
        humid_data = data['humid'].to_numpy(np.float64)

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

            Rlw = self.sb_const * (self.opac*(airtemp_data[n-1])**4 - (T[n-1])**4)
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
        """

        # Extract data from input file
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
            print('+++ Constant time-step interpolation +++')
            print('Interpolated dataset has '+str(len(df_final))+' time steps with length '+str(round(np.mean(dt_list),3))+' s.')
            print('++++++++++++++++++++++++++++++++++++++++')

        return df_final, dt_list, idx_list[:-1]

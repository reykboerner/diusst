"""
cfl_interpolation.py
CFL-dependent time step interpolation  of dataset
"""

import numpy as np
import pandas as pd
from datetime import datetime
from scipy.interpolate import interp1d

from diusst_funcs import make_mesh

def cfl_interpolation(data, dz0, ngrid, a=0,b=1,k_eddy_max=2e-4,k_mol=1e-7, maxwind=10, z_f=10, save=None, verbose=True):
    z = make_mesh(dz0,ngrid,z_f=z_f)[0]
    dz = z[1:]-z[:-1]

    times = data['times'].to_numpy(np.float64)

    sst = data['sst'].to_numpy(np.float64)
    sst_err = data['sst_err'].to_numpy(np.float64)
    ftemp = data['ftemp'].to_numpy(np.float64)
    wind = data['wind'].to_numpy(np.float64)
    atemp = data['atemp'].to_numpy(np.float64)
    swrad = data['swrad'].to_numpy(np.float64)
    humid = data['humid'].to_numpy(np.float64)

    series = np.stack((sst, sst_err, ftemp, wind, atemp, swrad, humid))
    times_concat = np.zeros(1)
    series_concat = np.zeros((7,1))
    dt_list = []

    for i in range(1,len(times)):
        # get wind speed
        u = min(maxwind,max(wind[i],wind[i-1]))
        # calculate courant for time step 1s at each depth
        c_array = 2*(k_mol+k_eddy_max*(a+b*np.abs(z[1:])/z_f))*u**2 / dz**2
        # get maximum CFL
        c = np.amax(c_array)
        # set time step
        dt = min(10, 0.95/c)
        # interpolate
        times_new = np.arange(times[i-1], times[i], dt)
        f = interp1d(times[i-1:i+1], series[:,i-1:i+1])
        series_new = f(times_new)

        times_concat = np.concatenate((times_concat,times_new))
        series_concat = np.column_stack((series_concat,series_new))
        dt_list.append(dt)

    final = np.row_stack((times_concat,series_concat)).transpose()
    df_final = pd.DataFrame(final[1:,:], columns=['times','sst','sst_err','ftemp','wind','atemp','swrad','humid'])

    if save is not None:
        df_final.to_csv(save+'_data.csv')

    if verbose:
        print('+++ Variable time-step interpolation +++')
        print('Interpolated dataset has '+str(len(df_final))+' time steps with average length '+str(round(np.mean(dt_list),3))+' s.')
        print('Constant dt interpolation would require dt = '+str(round(np.amin(dt_list)/0.95,3))+' s --> '+str(int((times[-1]-times[0])/np.amin(dt_list)/0.95))+' steps.')
        print('Computation time will be reduced by '+str(round((1-len(df_final)/int((times[-1]-times[0])/np.amin(dt_list)/0.95))*100,3))+' %')
        print('++++++++++++++++++++++++++++++++++++++++')

    return df_final, dt_list

#############################################################################################################

def cfl_interpolation5(data, dz0, ngrid, k_mol=1e-7, k_eddy_max=2e-4, k_0_max=1, lambd_min=0.5, maxwind=10, z_f=10, save=None, verbose=True):
    z = make_mesh(dz0,ngrid,z_f=z_f)[0]
    dz = z[1:]-z[:-1]

    times = data['times'].to_numpy(np.float64)

    sst = data['sst'].to_numpy(np.float64)
    sst_err = data['sst_err'].to_numpy(np.float64)
    ftemp = data['ftemp'].to_numpy(np.float64)
    wind = data['wind'].to_numpy(np.float64)
    atemp = data['atemp'].to_numpy(np.float64)
    swrad = data['swrad'].to_numpy(np.float64)
    humid = data['humid'].to_numpy(np.float64)

    series = np.stack((sst, sst_err, ftemp, wind, atemp, swrad, humid))
    times_concat = np.zeros(1)
    series_concat = np.zeros((7,1))
    dt_list = []

    for i in range(1,len(times)):
        # get wind speed
        u = min(maxwind,max(wind[i],wind[i-1]))
        # calculate courant for time step 1s at each depth
        c_array = 2*(k_mol+k_eddy_max*( 1-k_0_max*np.exp(z[1:]/lambd_min))/(1-k_0_max*np.exp(-z_f/lambd_min))*u**2 ) / dz**2
        # get maximum CFL
        c = np.amax(c_array)
        # set time step
        dt = min(10, 0.95/c)
        # interpolate
        times_new = np.arange(times[i-1], times[i], dt)
        f = interp1d(times[i-1:i+1], series[:,i-1:i+1])
        series_new = f(times_new)

        times_concat = np.concatenate((times_concat,times_new))
        series_concat = np.column_stack((series_concat,series_new))
        dt_list.append(dt)

    final = np.row_stack((times_concat,series_concat)).transpose()
    df_final = pd.DataFrame(final[1:,:], columns=['times','sst','sst_err','ftemp','wind','atemp','swrad','humid'])

    if save is not None:
        df_final.to_csv(save+'_data.csv')

    if verbose:
        print('+++ Variable time-step interpolation +++')
        print('Interpolated dataset has '+str(len(df_final))+' time steps with average length '+str(round(np.mean(dt_list),3))+' s.')
        print('Constant dt interpolation would require dt = '+str(round(np.amin(dt_list)/0.95,3))+' s --> '+str(int((times[-1]-times[0])/np.amin(dt_list)/0.95))+' steps.')
        print('Computation time will be reduced by '+str(round((1-len(df_final)/int((times[-1]-times[0])/np.amin(dt_list)/0.95))*100,3))+' %')
        print('++++++++++++++++++++++++++++++++++++++++')

    return df_final, dt_list



#############################################################################################################

def interpolate(data, dt, method='linear', save='False'):

    times = data['times'].to_numpy(np.float64)
    times_new = np.arange(times[0], times[-1], dt)

    sst = data['sst'].to_numpy(np.float64)
    sst_err = data['sst_err'].to_numpy(np.float64)
    ftemp = data['ftemp'].to_numpy(np.float64)
    wind = data['wind'].to_numpy(np.float64)
    atemp = data['atemp'].to_numpy(np.float64)
    swrad = data['swrad'].to_numpy(np.float64)
    humid = data['humid'].to_numpy(np.float64)

    series = np.stack((sst, sst_err, ftemp, wind, atemp, swrad, humid))

    if method == 'linear':
        f = interp1d(times,series)
        series_new = f(times_new)

    elif method == 'spline':
        print('not implemented yet')

    data_new = np.row_stack((times_new, series_new)).transpose()
    df_new = pd.DataFrame(data_new, columns=['times','sst','sst_err','ftemp','wind','atemp','swrad','humid'])

    if save:
        df_new.to_csv('interpolated_data.csv')

    return df_new

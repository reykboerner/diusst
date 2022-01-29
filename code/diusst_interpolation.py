"""
diusst_interpolation.py
CFL-dependent time step interpolation  of dataset
"""

# Load required modules
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.interpolate import interp1d

# Load custom functions
from diusst_funcs import make_mesh

def cfl_interpolation(data,
    dz0 = 0.1,
    ngrid = 40,
    z_f = 10,
    k_mol = 1e-7,
    kappa = 2e-4,
    sigma = 0.5,
    lambd = 3,
    wind_max = 10,
    CFL_max = 0.95,
    diffu_type = None,
    save=None,
    verbose=True):

    # Generate stretched mesh
    z = make_mesh(dz0,ngrid,z_f=z_f)[0]
    dz = (z[1:]-z[:-1])[:-1]

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

        # Get wind speed
        u = min(wind_max,max(wind[i],wind[i-1]))

        # Calculate CFL/dt at all grid points
        if diffu_type == 'BASE' or diffu_type == 'STAB' or diffu_type == 'STAB2':
            c_array = 2 * (k_mol + kappa*u**2) / dz**2

        elif diffu_type == 'LIN':
            c_array = 2 * (k_mol + kappa*u**2 * np.abs(z[1:-1]/z_f)) / dz**2

        elif diffu_type == 'LIN2':
            c_array = 2 * (k_mol + kappa*u**2 * (1 + sigma*(np.abs(z[1:-1]/z_f)-1))) / dz**2

        elif diffu_type == 'EXP':
            c_array = 2 * (k_mol + kappa*u**2 * (1-sigma*np.exp(z[1:-1]/lambd))/(1-sigma*np.exp(-z_f/lambd)) ) / dz**2

        else:
            print('Error: diffu_type not specified.')
            break

        # Get maximum CFL and compute time step dt
        CFL = np.amax(c_array)
        dt = min(10, CFL_max/CFL)

        # Interpolate
        times_new = np.arange(times[i-1], times[i], dt)
        f = interp1d(times[i-1:i+1], series[:,i-1:i+1])
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
        print('+++ Variable time-step interpolation +++')
        print('Interpolated dataset has '+str(len(df_final))+' time steps with average length '+str(round(np.mean(dt_list),3))+' s.')
        print('Constant dt interpolation would require dt = '+str(round(np.amin(dt_list),3))+' s --> '+str(int((times[-1]-times[0])/np.amin(dt_list)))+' steps.')
        print('Computation time will be reduced by '+str(round((1-len(df_final)/int((times[-1]-times[0])/np.amin(dt_list)))*100,3))+' %')
        print('++++++++++++++++++++++++++++++++++++++++')

    return df_final, dt_list, idx_list[:-1]

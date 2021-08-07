import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import time
sys.path.append('../code')

from interpolation import cfl_interpolation, cfl_interpolation5
from diusst_euler4S import diusst_eulerd

from diusst_funcs import make_mesh
from mscfunctions import fresnel

# Incoming SW radiation
def q_sw(t, a, reflect=1, n_a=1., n_w=1.34):
    if reflect == 1:
        return a * np.cos(2*np.pi*t/(60*60*24)+np.pi) * fresnel(2*np.pi*t/(60*60*24)+np.pi, n1=n_a, n2=n_w)
    elif reflect == 0:
        return a * np.cos(2*np.pi*t/(60*60*24)+np.pi)

# Boundary layer air temperature
def T_a(t, Tmax, Tmin, tau=0):
    return (Tmax+Tmin)/2 - (Tmax-Tmin)/2 * np.cos(2*np.pi/(60*60*24)*(t-tau))

#########################################################
def simulate_eulerd(params,
    dz0 = 0.10,
    ngrid = 40,
    z_f = 10,
    diffu = 1,
    opac = 1,
    k_mol = 1e-7,
    maxwind = 10,
    strat = 5,
    stratdepth=1,
    windstrength = 1
    ):

    times_orig = np.linspace(0,2*86400,2*24*60)
    swrad_orig = q_sw(times_orig,1)*1000
    atemp_orig = T_a(times_orig,301,299)
    wind_orig = 1+ windstrength + 1*np.cos(times_orig/86400*2*np.pi)
    humid_orig = np.ones(len(times_orig)) * 0.01
    ftemp_orig = np.ones(len(times_orig)) * 300
    sst_orig = np.zeros(len(times_orig))
    sst_err_orig = np.ones(len(times_orig)) * 0.1

    data_array = np.stack([times_orig, wind_orig, atemp_orig, swrad_orig, humid_orig,sst_orig,sst_err_orig,ftemp_orig])
    colnames = ['times','wind','atemp','swrad','humid','sst','sst_err','ftemp']
    data_orig = pd.DataFrame(data_array.transpose(),columns=colnames)

    k_eddy, mu, attenu, k0, gamma = params

    # interpolate to meet CFL condition
    data, dtlist, idx = cfl_interpolation5(data_orig, dz0=dz0, ngrid=ngrid,
        k_mol = k_mol,
        k_eddy_max=k_eddy, k_0_min=0, lambd_min=gamma,
        maxwind=maxwind, z_f=z_f)

    # extract data
    ftemp = np.mean(data['ftemp'].to_numpy(np.float64))
    sst_data = data['sst'].to_numpy(np.float64) - data['ftemp'].to_numpy(np.float64)
    sst_err = data['sst_err'].to_numpy(np.float64)
    times = data['times'].to_numpy(np.float64)
    wind = data['wind'].to_numpy(np.float64)
    atemp = data['atemp'].to_numpy(np.float64)
    atemp_rel = atemp - data['ftemp'].to_numpy(np.float64) + ftemp
    swrad = data['swrad'].to_numpy(np.float64)
    humid = data['humid'].to_numpy(np.float64)

    timer = time.time()
    sim = diusst_eulerd(
                    times, atemp_rel, swrad, u_data=wind, sa_data=humid, T_f=ftemp,
                    k_eddy=k_eddy, mu=mu, attenu=attenu, k_0=k0, lambd=gamma, strat=strat, stratdepth=stratdepth,
                    opac=opac, k_mol=k_mol,
                    dz=dz0, ngrid=ngrid)
    print('Done, took ', str(time.time()-timer))

    return sim

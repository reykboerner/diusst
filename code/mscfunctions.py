'''
Python functions for Reyk's MSc thesis
Last edited: 25. March 2021
'''

# import required packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import inv
from scipy import stats
from datetime import datetime

from diusst_funcs import make_mesh
#from iminuit import Minuit

def gaussian(x,mu,sigma):
	return np.exp(-(x-mu)**2/(2*sigma**2)) / np.sqrt(2*np.pi*sigma**2)

from scipy.interpolate import interp1d, splev, splrep
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


####################################################

def smart_interpolation(data, dz0, Ngrid, a=0,b=1,k_eddy_max=2e-4,k_mol=1e-7, save=False, verbose=True):

	z = make_mesh(dz0,Ngrid)[0]
	dz = z[1:]-z[:-1]

	times = data['times'].to_numpy(np.float64) * 86400

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
		u = max(wind[i],wind[i-1])
        # calculate courant for time step 1s at each depth
		c_array = 2*(k_mol+k_eddy_max*(a+b*np.abs(z[1:])))*u**2 / dz**2
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

	if save:
		df_final.to_csv(datetime.now().isoformat(timespec='seconds')+'smartinterpol_data.csv')

	if verbose:
		print('+++ Variable time-step interpolation +++')
		print('Interpolated dataset has '+str(len(df_final))+' time steps.')
		print('The average time step is '+str(round(np.mean(dt_list),3))+' seconds.')
		print('The minimum time step required for a stable run is '+str(round(np.amin(dt_list)/0.95,3))+' s --> '+str(int((times[-1]-times[0])/np.amin(dt_list)/0.95))+' steps required.')
		print('Computation time will be reduced by '+str(round((1-len(dt_list)/int((times[-1]-times[0])/np.amin(dt_list)/0.95))*100,3))+' %')

	return df_final, dt_list



####################################################


# Stretched grid functions
def Z(x, dz0, eps):
    return dz0 - dz0*(1-eps**x)/(1-eps)

def Z_inv(z, dz0, eps):
    z0 = dz0/2
    return np.log(1-(z0-z)*(1-eps)/dz0)/np.log(eps)

def dxdz(z, dz0, eps):
    return (1-eps)/((1+eps)*dz0 + 2*(1-eps)*z)

def dxdz2(z, dz0, eps):
    return - 4*(1-eps)**2/((1+eps)*dz0 + 2*(1-eps)*z)**2

# Transmittance based on Fresnel eqs
def fresnel(theta,n1=1.,n2=1.34):
    theta = theta % (2*np.pi)
    theta = np.where((theta > (np.pi/2)) & (theta < (3*np.pi/2)), np.pi/2, theta)

    Rs = ((n1*np.cos(theta) - n2*np.sqrt(1-(n1/n2*np.sin(theta))**2))/(n1*np.cos(theta) + n2*np.sqrt(1-(n1/n2*np.sin(theta))**2)))**2
    Rp = ( (n1*np.sqrt(1-(n1/n2*np.sin(theta))**2) - n2*np.cos(theta)) / (n1*np.sqrt(1-(n1/n2*np.sin(theta))**2) + n2*np.cos(theta)) )**2

    return 1 - (Rs+Rp)/2

# Refraction angle based on Snell's law
def snell(theta, n1=1., n2=1.34):
    theta = theta % (2*np.pi)
    theta = np.where((theta > (np.pi/2)) & (theta < (3*np.pi/2)), np.pi/2, theta)

    return np.abs(np.arcsin(n1/n2*np.sin(theta)))

#compute p percentile penetration depth of SW solar radiation
def sw_depth_at_centile(attenu, p=0.99, dz=None):
    if dz:
        return - np.log(1-p)/(attenu*dz)
    else:
        return - np.log(1-p)/attenu

# compute percentile p of SW radiation absorbed within depth z
def sw_centile_at_depth(z, attenu=0.5):
    return np.exp(-z*attenu)

# compute saturation specific humidity from temperature
def s_sat(T, rho_a, R_v):
    return 100 * 6.112 * np.exp( 17.67*(T-273.15)/(T-273.15+243.5) ) / (rho_a * R_v * T)

# Find elements of a dense sequence that are similar to the values of a coarse sequence
def get_near_indices(a,b):
    """
    Get indexes of values in array b that are nearest to the values in array a
    (The values in a and b must increase monotonically!)
    """
    c = np.zeros(a.shape[0])
    j = 0
    for i in range(a.shape[0]):
        while b[j] < a[i]:
            j += 1
        if np.abs(b[j-1]-a[i]) < np.abs(b[j]-a[i]):
            c[i] = j-1
        else:
            c[i] = j
    return c.astype(int)

# convert minutes since "start" into local sun time in hours
def utcmin_to_localhour(array,lon,start):
	"""
	array: the array of times, in units minute since "start"
	lon: longitude (east)
	start: start time in the integer format HHMMSS
	"""
	local_start = (np.floor(start/1e4) + lon/15) % 24 + (start/1e4-np.floor(start/1e4)) / 0.6
	local_array = (array/60 + local_start) % 24
	return local_array

# shortwave radiation function
def q_sw(t, a, reflect=1, n_a=1., n_w=1.34):
    if reflect == 1:
        return a * np.cos(2*np.pi*t/(60*60*24)+np.pi) * fresnel(2*np.pi*t/(60*60*24)+np.pi, n1=n_a, n2=n_w)
    elif reflect == 0:
        return a * np.cos(2*np.pi*t/(60*60*24)+np.pi)

#################################################
#SST model
#################################################

# v2 means that 'diffu' is multiplied to (k_mol + k_eddy) instead of just to k_eddy
def diusst_v2(Ta_data, sw_data, t_sim, dt,
    z_col=10,
    dz=0.02,
    sa_data=None,
    u_data=None,
    u=0,
    T_0=300,
    cloud=0,
    diffu=1.,
    mixing=0.04/3/60,
    sens=1.,
    latent=1.,
    attenu=0.8,
    opac=1.,
    k_mol = 1e-7,
    k_eddy = 1e-4,
    n_w = 1.34,
    n_a = 1.0,
    rho_w = 1027,
    rho_a = 1.1,
    s_a = 10*1e-3,                  # specific humidity at reference height above sea level (kg/kg)
    C_s = 1.3e-3,                   # turbulent exchange coefficient for sensible heat
    C_l = 1.5e-3,                   # turbulent exchange coefficient for latent heat
    L_evap = 2.5e6,                 # latent heat of evaporization (J/kg) Source: Klose p.151
    c_p = 3850,                     # specific heat of seawater at const pressure
    c_p_a = 1005,
    stefan_boltzmann = 5.67e-8,     # stefan boltzmann constant
    R_v = 461.51,                   # gas constant of water vapor, J/(kg*K)):
    output=1):

    # Define domain
    N_z = int(z_col / dz) + 2
    N_t = int(t_sim / dt)
    N = N_z - 2
    z = - np.arange(0, z_col + 2*dz, dz) + dz/2
    t = np.arange(0,t_sim, dt)

    # Atmospheric data
    if sa_data is None:
        sa_data = np.ones(N_t) * s_a

    if u_data is None:
        u_data = np.ones(N_t) * u

    # Initialize temperature array
    init = np.ones(N_z)*T_0
    T = np.array([init[1:-1]])

    #initialize integration matrix
    kappa  = diffu*(k_mol + k_eddy*u**2) #diffu*k_eddy*u**2
    alpha  = 1 + mixing*dt + 2*kappa * dt / dz**2
    beta   = - kappa * dt / dz**2
    gamma  = dt / (dz * rho_w * c_p)

    row1, row2, row3 = np.arange(N), np.arange(N-1), np.arange(1,N)
    ent1, ent2, ent3 = np.ones(N)*alpha, np.ones(N-1)*beta, np.ones(N-1)*beta
    #ent1[0] = 1 + mu + 2*(1.998*k_mol) * dt / dz**2
    #ent2[0] = - (1.998*k_mol) * dt / dz**2

    row = np.concatenate((row1, row2, row3))
    col = np.concatenate((row1, row3, row2))
    ent = np.concatenate((ent1, ent2, ent3))

    A = sparse.csc_matrix((ent, (row,col)), shape=(N,N))
    A_inv = inv(A).toarray()

    Qs, Ql, Rlw, Rsw = [], [], [], []

    for n in range(1, N_t):

        solangle = 2*np.pi*n*dt/(60*60*24)+np.pi

        # air-sea fluxes into ocean (positive downwards)
        R_sw = (1-cloud) * sw_data[n] * np.exp(attenu/np.cos(snell(solangle))*z)
        R_lw = stefan_boltzmann * (opac*(Ta_data[n])**4 - (T[-1,1])**4)
        Q_s  = sens* rho_a * c_p_a * C_s * max(0.5, u_data[n]) * (Ta_data[n] - T[-1,1])
        Q_l  = latent * rho_a * L_evap * C_l * max(0.5, u_data[n]) * (sa_data[n] - s_sat(T[-1,1], rho_a, R_v))

        # solution vector
        b     = gamma * (R_sw[:-2] - R_sw[1:-1]) + mixing*dt*T_0
        b[0] += gamma * (R_lw + Q_l + Q_s) - beta * T[-1,0]
        b[-1] = - (beta) * T_0 + mixing*dt*T_0

        Tn = np.dot(A_inv, (T[n-1] + b))

        T = np.concatenate((T, np.array([Tn])), axis=0)
        Qs.append(Q_s)
        Ql.append(Q_l)
        Rlw.append(R_lw)
        Rsw.append(R_sw[0])

    # Output data
    t_hour = 60*60
    t_day = t_hour*24

    skin_diff = T[:,0] - T_0
    colmax_diff = np.amax(T, axis=1) - T_0
    colmax_depth = np.argmax(T,axis=1)*dz

    if output==1:
        return [T, z, t, Qs, Ql, Rlw, Rsw]
    elif output==2:
        return [skin_diff, t, [Qs, Ql, Rlw, Rsw]]
    elif output==3:
        return [colmax_diff, t, [Qs, Ql, Rlw, Rsw], colmax_depth]


# v3 is like v2 (above) but allows variable grid spacing in the depth dimension
def diusst_v3(Ta_data, sw_data, t_sim, dt,
    z_col=10,
    dz=np.ones(int(10/0.02)+1)*0.02,
    sa_data=None,
    u_data=None,
    u=0,
    T_0=300,
    cloud=0,
    diffu=1.,
    mixing=0.04/3/60,
    sens=1.,
    latent=1.,
    attenu=0.8,
    opac=1.,
    k_mol = 1e-7,
    k_eddy = 1e-4,
    n_w = 1.34,
    n_a = 1.0,
    rho_w = 1027,
    rho_a = 1.1,
    s_a = 10*1e-3,                  # specific humidity at reference height above sea level (kg/kg)
    C_s = 1.3e-3,                   # turbulent exchange coefficient for sensible heat
    C_l = 1.5e-3,                   # turbulent exchange coefficient for latent heat
    L_evap = 2.5e6,                 # latent heat of evaporization (J/kg) Source: Klose p.151
    c_p = 3850,                     # specific heat of seawater at const pressure
    c_p_a = 1005,
    stefan_boltzmann = 5.67e-8,     # stefan boltzmann constant
    R_v = 461.51,                   # gas constant of water vapor, J/(kg*K)):
    output=1):

    # Define domain
    N_z = dz.shape[0] + 1
    N_t = int(t_sim / dt)
    N = N_z - 2
    #z = - np.arange(0, z_col + 2*dz, dz) + dz/2
    z = np.ones(N_z)*dz[0]/2
    for i in range(1,N_z):
        z[i] = z[i-1] - dz[i-1]
    t = np.arange(0,t_sim, dt)

    # Atmospheric data
    if sa_data is None:
        sa_data = np.ones(N_t) * s_a

    if u_data is None:
        u_data = np.ones(N_t) * u

    # Initialize temperature array
    init = np.ones(N_z)*T_0
    T = np.array([init[1:-1]])

    #initialize integration matrix
    kappa  = diffu*(k_mol + k_eddy*u**2) #diffu*k_eddy*u**2

    alpha, beta, gamma = np.zeros(N), np.zeros(N), np.zeros(N)
    for i in range(N):
        alpha[i] = 1 + mixing*dt + 2*kappa * dt / (dz[i]*dz[i+1])
        beta[i]   = - kappa * dt / (dz[i]*dz[i+1])
        gamma[i]  = dt / (dz[i] * rho_w * c_p)

    row1, row2, row3 = np.arange(N), np.arange(N-1), np.arange(1,N)
    ent1, ent2, ent3 = alpha, beta[:-1], beta[1:]
    #ent1[0] = 1 + mu + 2*(1.998*k_mol) * dt / dz**2
    #ent2[0] = - (1.998*k_mol) * dt / dz**2

    row = np.concatenate((row1, row2, row3))
    col = np.concatenate((row1, row3, row2))
    ent = np.concatenate((ent1, ent2, ent3))

    A = sparse.csc_matrix((ent, (row,col)), shape=(N,N))
    A_inv = inv(A).toarray()

    Qs, Ql, Rlw, Rsw = [], [], [], []

    for n in range(1, N_t):

        solangle = 2*np.pi*n*dt/(60*60*24)+np.pi

        # air-sea fluxes into ocean (positive downwards)
        R_sw = (1-cloud) * sw_data[n] * np.exp(attenu/np.cos(snell(solangle))*z)
        R_lw = stefan_boltzmann * (opac*(Ta_data[n])**4 - (T[-1,1])**4)
        Q_s  = sens* rho_a * c_p_a * C_s * max(0.5, u_data[n]) * (Ta_data[n] - T[-1,1])
        Q_l  = latent * rho_a * L_evap * C_l * max(0.5, u_data[n]) * (sa_data[n] - s_sat(T[-1,1], rho_a, R_v))

        # solution vector
        b     = gamma * (R_sw[:-2] - R_sw[1:-1]) + mixing*dt*T_0
        b[0] += gamma[0] * (R_lw + Q_l + Q_s) - beta[0] * T[-1,0]
        b[-1] = - beta[-1] * T_0 + mixing*dt*T_0

        Tn = np.dot(A_inv, (T[n-1] + b))

        T = np.concatenate((T, np.array([Tn])), axis=0)
        Qs.append(Q_s)
        Ql.append(Q_l)
        Rlw.append(R_lw)
        Rsw.append(R_sw[0])

    # Output data
    t_hour = 60*60
    t_day = t_hour*24

    skin_diff = T[:,0] - T_0
    colmax_diff = np.amax(T, axis=1) - T_0
    #colmax_depth = np.argmax(T,axis=1)*dz

    if output==1:
        return [T, z, t, Qs, Ql, Rlw, Rsw]
    elif output==2:
        return [skin_diff, t, [Qs, Ql, Rlw, Rsw]]
    elif output==3:
        return [colmax_diff, t, [Qs, Ql, Rlw, Rsw], colmax_depth]

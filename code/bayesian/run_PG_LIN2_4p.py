"""
run_PG_LIN2_4p.py
Python script to run Bayesian analysis for DIUSST model
"""

###########################################################
# RUN SETTINGS (check before each run)

# Output storage
run_id = 'PG_LIN2_4p-A1'
output_path = '../../output_files/'

# Fit parameters
param_names = ['kappa', 'mu', 'alpha', 'sigma']

# Parameter limits
param_min = [0, 0.0005, 0.05, 0]
param_max = [5e-4, 0.05, 10, 1]

# Initial walker positions
param_start = [1e-4, 0.006, 4, 0.8]

# Sampling
nwalkers = 24
nsteps = int(1e4)

# Prior distribution for mu
mu0 = 0.006
mustd = 0.0015

# DIUSST model
diffu_type = 'LIN2'
z_f = 10
dz = 0.10
ngrid = 40
#sigma = 0.8
lambd = 3
z_ref = 1
k_mol = 1e-7
wind_max = 10
CFL = 0.95
ref_level = int(21)

data_path = '../../input_data/moce5/'
data_filename = 'training_moce5_err-boatspd-x2_humid10.csv'
data_interval1 = [96,413]
data_interval2 = [1290,1585]
removeidx2 = [69,74,81,82,99,100,171,172,176]

# Other settings
parallel = True
use_backend = True

###########################################################
# (End of run settings)
###########################################################

# Change working directory to current one
import os
import sys
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
parent = os.path.dirname(dname)
os.chdir(dname)
sys.path.append(parent)


# Load external modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import emcee
import corner
from multiprocessing import Pool
from multiprocessing import cpu_count

# Load custom functions
from diusst_interpolation import cfl_interpolation
from diusst_funcs import make_mesh
from diusst_model import diusst

# Time stamp
from datetime import datetime
now = datetime.now()
timestamp = now.strftime("%y%m%d-%H%M%S")

# Initialize
param_min = np.array(param_min)
param_max = np.array(param_max)
param_start = np.array(param_start)

print('==== Bayesian sampling {} parameters, Run ID '.format(len(param_start))+run_id+' ====')
print('Start time: '+str(timestamp))
print('Data loaded from '+data_path+data_filename)
print('Optimization with respect to reference depth = '+str(round(make_mesh(0.1,40,z_f=10)[0][ref_level],3)))

###########################################################
# Load dataset
data_orig1 = pd.read_csv(data_path+data_filename)[data_interval1[0]:data_interval1[1]]

data_orig2_dirty = pd.read_csv(data_path+data_filename)[data_interval2[0]:data_interval2[1]]
data_orig2 = data_orig2_dirty.drop(data_orig2_dirty.index[removeidx2])

# Extract data from dataset 1
ftemp1 = np.mean(data_orig1['ftemp'].to_numpy(np.float64))

times_orig1 = data_orig1['times'].to_numpy(np.float64)
sst_data1 = data_orig1['sst'].to_numpy(np.float64) - data_orig1['ftemp'].to_numpy(np.float64)
sst_err1 = data_orig1['sst_err'].to_numpy(np.float64)

# Extract data from dataset 2
ftemp2 = np.mean(data_orig2['ftemp'].to_numpy(np.float64))

times_orig2 = data_orig2['times'].to_numpy(np.float64)
sst_data2 = data_orig2['sst'].to_numpy(np.float64) - data_orig2['ftemp'].to_numpy(np.float64)
sst_err2 = data_orig2['sst_err'].to_numpy(np.float64)

###########################################################
# Define likelihood function
def bayesian_likelihood(params):
    kappa, mu, alpha, sigma = params

    # interpolate to meet CFL condition
    data1, dtlist1, idx1 = cfl_interpolation(data_orig1, dz0=dz, ngrid=ngrid, z_f=z_f,
            k_mol = k_mol, kappa=kappa, sigma=sigma, lambd=lambd,
            wind_max=wind_max, CFL_max=CFL, diffu_type=diffu_type,
            save=None,verbose=False)

    data2, dtlist2, idx2 = cfl_interpolation(data_orig2, dz0=dz, ngrid=ngrid, z_f=z_f,
            k_mol = k_mol, kappa=kappa, sigma=sigma, lambd=lambd,
            wind_max=wind_max, CFL_max=CFL, diffu_type=diffu_type,
            save=None,verbose=False)

    times1 = data1['times'].to_numpy(np.float64)
    wind1 = data1['wind'].to_numpy(np.float64)
    swrad1 = data1['swrad'].to_numpy(np.float64)
    humid1 = data1['humid'].to_numpy(np.float64)
    atemp_rel1 = data1['atemp'].to_numpy(np.float64) - data1['ftemp'].to_numpy(np.float64) + ftemp1

    times2 = data2['times'].to_numpy(np.float64)
    wind2 = data2['wind'].to_numpy(np.float64)
    swrad2 = data2['swrad'].to_numpy(np.float64)
    humid2 = data2['humid'].to_numpy(np.float64)
    atemp_rel2 = data2['atemp'].to_numpy(np.float64) - data2['ftemp'].to_numpy(np.float64) + ftemp2

    simu1 = diusst(
            times1, wind1, swrad1, atemp_rel1, humid1, T_f=ftemp1,
            kappa=kappa, mu=mu, alpha=alpha, lambd=lambd, sigma=sigma, z_ref=z_ref,
            diffu_type=diffu_type, wind_max=wind_max, k_mol=k_mol, z_f=z_f, dz=dz, ngrid=ngrid,
            output='temp')

    simu2 = diusst(
            times2, wind2, swrad2, atemp_rel2, humid2, T_f=ftemp2,
            kappa=kappa, mu=mu, alpha=alpha, lambd=lambd, sigma=sigma, z_ref=z_ref,
            diffu_type=diffu_type, wind_max=wind_max, k_mol=k_mol, z_f=z_f, dz=dz, ngrid=ngrid,
            output='temp')

    sst_model1 = simu1[:,0]-simu1[:,ref_level]
    sst_model2 = simu2[:,0]-simu2[:,ref_level]

    sum1 = np.sum( (sst_model1[idx1] - sst_data1[:-1])**2 / sst_err1[:-1]**2 )
    sum2 = np.sum( (sst_model2[idx2] - sst_data2[:-1])**2 / sst_err2[:-1]**2 )
    mse = sum1 + sum2
    return mse

###########################################################
# Define posterior distribution function
def log_prob(x):
    if (x<param_min).any() or (x>param_max).any():
        return -np.inf
    else:
        mse = bayesian_likelihood(x)
        log_prior = - (x[1]-mu0)**2/(2*mustd**2) - np.log( (param_max[0]-param_min[0])*(param_max[2]-param_min[2])*(param_max[3]-param_min[3]) )
        return - mse/100 + log_prior

###########################################################
# Initialize emcee
ndim = len(param_names)
initial = np.zeros((nwalkers, ndim))
for i in range(nwalkers):
    initial[i] = param_start + 0.1*param_start*np.random.uniform(size=ndim)

# Prerequisites for parallelization
os.environ["OMP_NUM_THREADS"] = "1"
ncpu = cpu_count()

print('Ndim = {0} parameters, sample = {1} steps.'.format(ndim, nsteps))
print("{} walkers on {} CPUs.".format(nwalkers, ncpu))

# Backend
backend_filename = timestamp+'_'+run_id+'.h5'
backend = emcee.backends.HDFBackend(backend_filename)
backend.reset(nwalkers, ndim)

# run MCMC sampler
if parallel:
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, pool=pool, backend=backend)
        mcrun = sampler.run_mcmc(initial, nsteps, progress=True)

else:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
    mcrun = sampler.run_mcmc(initial, nsteps, progress=True)

print('Sampling complete ({} steps).'.format(nsteps))

# Get sampling results
samples = sampler.get_chain()
autocorrtime = sampler.get_autocorr_time(quiet=True)
probs = sampler.get_log_prob()

for i in range(ndim):
	np.savetxt(output_path+timestamp+'_'+run_id+'_samples_'+param_names[i]+'.txt', samples[:,:,i])

np.savetxt(output_path+timestamp+'_'+run_id+'_autocorrtime.txt', autocorrtime)
np.savetxt(output_path+timestamp+'_'+run_id+'_probs.txt', probs)

print('Sample storage complete.')
print('=================================== Success! YIPPIYAYEAH!!!')

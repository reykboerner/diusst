"""
run_3p_both.py
Python script to run Bayesian analysis for DIUSST model
"""

# Change working directory to curr
import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# Load external modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import emcee
import corner
from multiprocessing import Pool
from multiprocessing import cpu_count

# Load custom functions
from interpolation import cfl_interpolation5
from diusst_funcs import make_mesh
from diusst_model5 import diusst_bayesian as diusst

# Time stamp
from datetime import datetime
now = datetime.now()
timestamp = now.strftime("%y%m%d-%H%M%S")

###########################################################
# RUN SETTINGS (check before each run)

# Output storage
run_id = '3p-both-A1'
output_path = '../output/'

# Fit parameters
param_names = ['kappa', 'mu', 'attenu']

# Parameter limits
param_min = np.array([0, 0.0005, 0.05])
param_max = np.array([5e-4, 0.05, 10])

# Initial walker positions
param_start = np.array([1e-4, 0.006, 4])

# Sampling
nwalkers = 32
nsteps = 5000

# Prior
mu0 = 0.006
mustd = 0.0015

# DIUSST model
k0 = 0.8
lambd = 3
z_f = 10
dz0 = 0.10
ngrid = 40
diffu = 1
opac = 1
k_mol = 1e-7
maxwind = 10
ref_level = int(21)

data_path = '../data/bayesian_training/'
data_filename = 'training_minnett_err-boatspd-x2_humid10.csv'
data_interval1 = [96,413]
data_interval2 = [1290,1585]
removeidx2 = [69,74,81,82,99,100,171,172,176]

# Other settings
parallel = True
use_backend = True

###########################################################
# (End of run settings)
###########################################################

print('==== Bayesian sampling 3 parameters, Run ID '+run_id+' ====')
print('Start time: '+str(timestamp))
print('Data loaded from '+data_path+data_filename)
print('Optimization with respect to reference depth = '+str(round(make_mesh(0.1,40,z_f=10)[0][ref_level],3)))

# Load dataset
data_orig1 = pd.read_csv(data_path+data_filename)[data_interval1[0]:data_interval1[1]]

data_orig2_dirty = pd.read_csv(data_path+data_filename)[data_interval2[0]:data_interval2[1]]
data_orig2 = data_orig2_dirty.drop(data_orig2_dirty.index[removeidx2])


# extract data from dataset 1
ftemp1 = np.mean(data_orig1['ftemp'].to_numpy(np.float64))

times_orig1 = data_orig1['times'].to_numpy(np.float64)
sst_data1 = data_orig1['sst'].to_numpy(np.float64) - data_orig1['ftemp'].to_numpy(np.float64)
sst_err1 = data_orig1['sst_err'].to_numpy(np.float64)

# extract data from dataset 2
ftemp2 = np.mean(data_orig2['ftemp'].to_numpy(np.float64))

times_orig2 = data_orig2['times'].to_numpy(np.float64)
sst_data2 = data_orig2['sst'].to_numpy(np.float64) - data_orig2['ftemp'].to_numpy(np.float64)
sst_err2 = data_orig2['sst_err'].to_numpy(np.float64)

# Define likelihood function
def bayesian_likelihood(params):
    kappa, mu, attenu = params

    # interpolate to meet CFL condition
    data1, dtlist1, idx1 = cfl_interpolation5(data_orig1, dz0=dz0, ngrid=ngrid,
            k_mol = k_mol,
            k_eddy_max=kappa, k_0_min=k0, lambd_min=lambd,
            maxwind=maxwind, z_f=z_f,
            save=None,verbose=False)

    data2, dtlist2, idx2 = cfl_interpolation5(data_orig2, dz0=dz0, ngrid=ngrid,
            k_mol = k_mol,
            k_eddy_max=kappa, k_0_min=k0, lambd_min=lambd,
            maxwind=maxwind, z_f=z_f,
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
            times1, atemp_rel1, swrad1, u_data=wind1, sa_data=humid1, T_f=ftemp1,
            k_eddy=kappa, mu=mu, attenu=attenu, k_0=k0, lambd=lambd,
            opac=opac, k_mol=k_mol,
            dz=dz0, ngrid=ngrid)
    simu2 = diusst(
            times2, atemp_rel2, swrad2, u_data=wind2, sa_data=humid2, T_f=ftemp2,
            k_eddy=kappa, mu=mu, attenu=attenu, k_0=k0, lambd=lambd,
            opac=opac, k_mol=k_mol,
            dz=dz0, ngrid=ngrid)

    sst_model1 = simu1[:,0]-simu1[:,ref_level]
    sst_model2 = simu2[:,0]-simu2[:,ref_level]

    sum1 = np.sum( (sst_model1[idx1] - sst_data1[:-1])**2 / sst_err1[:-1]**2 )
    sum2 = np.sum( (sst_model2[idx2] - sst_data2[:-1])**2 / sst_err2[:-1]**2 )
    mse = sum1 + sum2
    return mse

# Define posterior distribution function
def log_prob(x):
    if (x<param_min).any() or (x>param_max).any():
        return -np.inf
    else:
        mse = bayesian_likelihood(x)
        log_prior = - (x[1]-mu0)**2/(2*mustd**2) - np.log( (param_max[0]-param_min[0])*(param_max[2]-param_min[2]) )
        return - mse/100 + log_prior

# initialize emcee
ndim = len(param_names)
initial = np.zeros((nwalkers, ndim))
for i in range(nwalkers):
    initial[i] = param_start + 0.1*param_start*np.random.uniform(size=ndim)

# prerequisites for parallelization
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

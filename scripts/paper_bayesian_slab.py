"""
run_PG_slab.py
Python script to run Bayesian analysis for DIUSST model
"""

###########################################################
# RUN SETTINGS (check before each run)

# Output storage
run_id = 'PG_paper_slab-A1'
output_path = '../../output_files/'

# Fit parameters
param_names = ['d', 'Q_sink', 'xi_1', 'xi_2']

# Parameter limits
param_min = [0.1, 0, 0, 0]
param_max = [10, 1000, 1, 1]

# Initial walker positions
param_start = [1, 100, 1e-4, 1e-9]

# Sampling
nwalkers = 24
nsteps = int(1e4)

# Prior distribution for Q_sink
sink0 = 100
sinkstd = 10

# Model
tstep = 3
wind_max = 10

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
from diusst_interpolation import cfl_interpolation, const_interpolation
from slab import Slab

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
    d, Q_sink, xi_1, xi_2 = params

    # interpolate to meet CFL condition
    data1, dtlist1, idx1 = const_interpolation(data_orig1, dt=tstep,
            save=None,verbose=False)

    data2, dtlist2, idx2 = const_interpolation(data_orig2, dt=tstep,
            save=None,verbose=False)

    data1['atemp'] = data1['atemp'] - data1['ftemp'] + ftemp1
    data2['atemp'] = data2['atemp'] - data2['ftemp'] + ftemp2

    atemp_rel1 = data1['atemp'].to_numpy(np.float64) - data1['ftemp'].to_numpy(np.float64) + ftemp1
    atemp_rel2 = data2['atemp'].to_numpy(np.float64) - data2['ftemp'].to_numpy(np.float64) + ftemp2

    simu1 = Slab(d=d, Q_sink=Q_sink, xi_1=xi_1, xi_2=xi_2, T_f=ftemp1).simulate(data1)
    simu2 = Slab(d=d, Q_sink=Q_sink, xi_1=xi_1, xi_2=xi_2, T_f=ftemp2).simulate(data2)

    sst_model1 = simu1 - ftemp1
    sst_model2 = simu2 - ftemp2

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
        log_prior = - (x[1]-sink0)**2/(2*sinkstd**2) - np.log( (param_max[0]-param_min[0])*(param_max[2]-param_min[2])*(param_max[3]-param_min[3]) )
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

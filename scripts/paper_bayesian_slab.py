__author__ = 'Reyk Boerner'

###########################################################
# RUN SETTINGS ############################################
###########################################################

# Output storage
run_id = 'paper_bayesian_slab_ftemp-totalmean'
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
const_humid = True

data_path = '../input_data/moce5/'
data_filename = 'moce5_dataset.cdf'
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
import os, sys
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
parent = os.path.dirname(dname)
os.chdir(dname)
sys.path.append(parent+'/src/')

# Load external modules
import numpy as np
import xarray as xr
import emcee
from multiprocessing import Pool, cpu_count

# Load custom functions
from slab import Slab

# Time stamp
from datetime import datetime
now = datetime.now()
timestamp = now.strftime("%y%m%d-%H%M%S")

# Initialize
param_min = np.array(param_min)
param_max = np.array(param_max)
param_start = np.array(param_start)

print('=== Bayesian sampling {} parameters, Run ID '.format(len(param_start))+run_id+' ===')
print('... Start time: {}'.format(now))
print('... Data loaded from '+data_path+data_filename)

###########################################################
# Load dataset
data_orig1 = xr.load_dataset(data_path+data_filename, decode_timedelta=False).isel(time=slice(data_interval1[0], data_interval1[1]))

data_orig2_dirty = xr.load_dataset(data_path+data_filename, decode_timedelta=False).isel(time=slice(data_interval2[0], data_interval2[1]))
data_orig2 = data_orig2_dirty.drop_isel(time=removeidx2)

# Get mean foundation temperature
ftemp = float(xr.load_dataset(data_path+data_filename, decode_timedelta=False).T_f)

# Extract data from dataset 1
times_orig1 = data_orig1['time'].to_numpy()
dsst_data1 = data_orig1['dsst'].to_numpy()
dsst_err1 = data_orig1['dsst_err'].to_numpy()

# Extract data from dataset 2
times_orig2 = data_orig2['time'].to_numpy()
dsst_data2 = data_orig2['dsst'].to_numpy()
dsst_err2 = data_orig2['dsst_err'].to_numpy()

print('... Foundation temperature set to {:.3f} K.'.format(ftemp))
if const_humid:
    print('... Specific humidity set to {:.1f} g/kg.'.format(float(data_orig1['humid'].isel(time=0))*1000))

###########################################################
# Define likelihood function
def bayesian_likelihood(params):
    d, Q_sink, xi_1, xi_2 = params

    model = Slab(T_f=ftemp, d=d, Q_sink=Q_sink, xi_1=xi_1, xi_2=xi_2)

    # interpolate to meet CFL condition
    data1, dtlist1, idx1 = model.interpolate(data_orig1, dt=tstep, verbose=False)
    data2, dtlist2, idx2 = model.interpolate(data_orig2, dt=tstep, verbose=False)

    simu1 = model.simulate(data1, progress=False)
    simu2 = model.simulate(data2, progress=False)

    sst_model1 = simu1 - ftemp
    sst_model2 = simu2 - ftemp

    sum1 = np.sum( (sst_model1[idx1] - dsst_data1[:-1])**2 / dsst_err1[:-1]**2 )
    sum2 = np.sum( (sst_model2[idx2] - dsst_data2[:-1])**2 / dsst_err2[:-1]**2 )
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

# Initialize emcee
ndim = len(param_names)
initial = np.zeros((nwalkers, ndim))
for i in range(nwalkers):
    initial[i] = param_start + 0.1*param_start*np.random.uniform(size=ndim)

# Prerequisites for parallelization
os.environ["OMP_NUM_THREADS"] = "1"
ncpu = cpu_count()

print('... Ndim = {0} parameters, sample = {1} steps.'.format(ndim, nsteps))
print("... {} walkers on {} CPUs.".format(nwalkers, ncpu))

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

print('... Sampling complete ({} steps).'.format(nsteps))

# Get sampling results
samples = sampler.get_chain()
autocorrtime = sampler.get_autocorr_time(quiet=True)
probs = sampler.get_log_prob()

for i in range(ndim):
	np.savetxt(output_path+timestamp+'_'+run_id+'_samples_'+param_names[i]+'.txt', samples[:,:,i])

np.savetxt(output_path+timestamp+'_'+run_id+'_autocorrtime.txt', autocorrtime)
np.savetxt(output_path+timestamp+'_'+run_id+'_probs.txt', probs)

print('... Sample storage complete.')
print('=================================== Success!')

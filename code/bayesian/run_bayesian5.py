"""
run_bayesian5.py
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
from diusst_model5 import diusst_bayesian as diusst

# Time stamp
from datetime import datetime
now = datetime.now()
timestamp = now.strftime("%y%m%d-%H%M%S")

###########################################################
# RUN SETTINGS (check before each run)

# Output storage
run_id = 'M-I6-B1-final'
output_path = '../output/'

# Fit parameters
param_names = ['kappa', 'mu', 'attenu', 'kappa0', 'lambd']

# Parameter limits
param_min = np.array([0, 0, 1e-3, 0.5, 3])
param_max = np.array([5e-4, 0.1, 10, 1, 10])

# Initial walker positions
param_start = np.array([1e-4, 0.005, 4, 0.8, 4])

# Sampling
nwalkers = 32
nsteps = 5000

# DIUSST model
scheme = 'euler'
z_f = 10
dz0 = 0.10
ngrid = 40
diffu = 1
opac = 1
k_mol = 1e-7
maxwind = 10

# Dataset
data_path = '../data/bayesian_training/'
data_filename = 'training_minnett_ssterr03-10_humid10.csv'
data_interval = [1149,1471]

# Other settings
parallel = True
use_backend = True

###########################################################
# (End of run settings)
###########################################################

print('==== Bayesian sampling 5 parameters, Run ID '+run_id+' ====')
print('Start time: '+str(timestamp))
print('Data loaded from '+data_path+data_filename)

# Load dataset
data_orig = pd.read_csv(data_path+data_filename)[data_interval[0]:data_interval[1]]

# interpolate to meet CFL condition
data, dtlist, idxlist = cfl_interpolation5(data_orig, dz0=dz0, ngrid=ngrid,
        k_mol = k_mol,
        k_eddy_max=param_max[0], k_0_max=param_max[3], lambd_min=param_min[-1],
        maxwind=maxwind, z_f=z_f,
        save=output_path+timestamp+'_'+run_id)

# extract data
ftemp = np.mean(data_orig['ftemp'].to_numpy(np.float64))

times = data['times'].to_numpy(np.float64)
wind = data['wind'].to_numpy(np.float64)
swrad = data['swrad'].to_numpy(np.float64)
humid = data['humid'].to_numpy(np.float64)
atemp_rel = data['atemp'].to_numpy(np.float64) - data['ftemp'].to_numpy(np.float64) + ftemp

times_orig = data_orig['times'].to_numpy(np.float64)
sst_data = data_orig['sst'].to_numpy(np.float64) - data_orig['ftemp'].to_numpy(np.float64) + ftemp
sst_err = data_orig['sst_err'].to_numpy(np.float64) *0.01


# Define likelihood function
def bayesian_likelihood(params):
    kappa, mu, attenu, kappa0, lambd = params
    simu = diusst(
            times, atemp_rel, swrad, u_data=wind, sa_data=humid, T_f=ftemp,
            k_eddy=kappa, mu=mu, attenu=attenu, k_0=kappa0, lambd=lambd,
            opac=opac, k_mol=k_mol,
            dz=dz0, ngrid=ngrid)
    sst_model = simu[0][:,0]
    print(len(sst_model[idxlist]),len(sst_data))
    mse = np.sum(((sst_model[idxlist] - sst_data[:-1])/sst_err[:-1])**2)
    return mse

# Define posterior distribution function
def log_prob(x):
    if (x<param_min).any() or (x>param_max).any():
        return -np.inf
    else:
        mse = bayesian_likelihood(x)
        return - (mse + np.log( np.prod(param_max-param_min) ))

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
print('=================================== Success!')
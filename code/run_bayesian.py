"""
run_bayesian.py
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
from interpolation import cfl_interpolation
from diusst_model import diusst_bayesian as diusst

# Time stamp
from datetime import datetime
now = datetime.now()
timestamp = now.strftime("%y%m%d-%H%M%S")

###########################################################
# RUN SETTINGS (check before each run)

# Output storage
output_path = '../output/'
run_id = 'M-I2-1'

# Fit parameters
param_names = ['kappa', 'mu', 'attenu']

# Parameter limits
param_min = np.array([0, 0, 0])
param_max = np.array([7e-4, 1, 10])

# Initial walker positions
param_start = np.array([2e-4, 1e-4, 2.5])

# Sampling
nwalkers = 16
nsteps = 700
burninsteps = 50

# DIUSST model
scheme = 'euler'
dz0 = 0.05
ngrid = 80
diffu = 1
opac = 1
k_mol = 1e-7
maxwind = 10

# Dataset
data_path = '../data/bayesian_training/'
data_filename = 'training_minnett_ssterr03-10_humid10.csv'
data_interval = [786,1377]

# Other settings
parallel = True

###########################################################
# (End of run settings)
###########################################################

print('==== Bayesian sampling, Run ID '+run_id+' ====')
print('Start time: '+str(timestamp))
print('Data loaded from '+data_path+data_filename)

# Load dataset
data_orig = pd.read_csv(data_path+data_filename)[data_interval[0]:data_interval[1]]

# interpolate to meet CFL condition
data = cfl_interpolation(data_orig, dz0=dz0, ngrid=ngrid,
        a=0, b=1, k_eddy_max=param_max[0], maxwind=maxwind)[0]

# extract data
ftemp = np.mean(data['ftemp'].to_numpy(np.float64))
sst_data = data['sst'].to_numpy(np.float64) - data['ftemp'].to_numpy(np.float64)
sst_err = data['sst_err'].to_numpy(np.float64)
times = data['times'].to_numpy(np.float64)
wind = data['wind'].to_numpy(np.float64)
atemp = data['atemp'].to_numpy(np.float64)
swrad = data['swrad'].to_numpy(np.float64)
humid = data['humid'].to_numpy(np.float64)

# Define likelihood function
def bayesian_likelihood(params):
    kappa, mu, attenu = params
    simu = diusst(
            times, atemp, swrad, u_data=wind, sa_data=humid, T_f=ftemp,
            k_eddy=kappa, mu=mu, attenu=attenu,
            opac=opac, k_mol=k_mol,
            dz=dz0, ngrid=ngrid)
    sst_model = simu[0][:,0] - ftemp
    mse = np.mean(((sst_data-sst_model)/sst_err)**2)
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

print('Ndim = {0} parameters, burn-in = {1} steps, followed by sampling {2} steps.'.format(ndim, burninsteps, nsteps))
print("{} walkers on {} CPUs.".format(nwalkers, ncpu))

# run MCMC sampler
if parallel:
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, pool=pool)
        burnin = sampler.run_mcmc(initial, burninsteps, progress=True)
        sampler.reset()
        mcrun = sampler.run_mcmc(burnin, nsteps, progress=True)

else:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
    burnin = sampler.run_mcmc(initial, burninsteps, progress=True)
    sampler.reset()
    mcrun = sampler.run_mcmc(burnin, nsteps, progress=True)

print('Sampling complete.')

# Get sampling results
samples = sampler.get_chain()
autocorrtime = sampler.get_autocorr_time()
probs = sampler.get_log_prob()

samples_df = pd.DataFrame(samples)
autocorrtime_df = pd.DataFrame(autocorrtime)
probs_df = pd.DataFrame(probs)

samples_df.to_csv(data_path+timestamp+'_'+run_id+'_samples.csv')
autocorrtime_df.to_csv(data_path+timestamp+'_'+run_id+'_autocorrtime.csv')
probs_df.to_csv(data_path+timestamp+run_id+'_'+'_probs.csv')

print('Sample storage complete.')
print('=================================== Success!')

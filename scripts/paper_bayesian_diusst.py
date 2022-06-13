__author__ = 'Reyk Boerner'

###########################################################
# RUN SETTINGS ############################################
###########################################################

# Output storage
run_id = 'paper_bayesian_diusst-A1_ftemp-totalmean'
output_path = '../../output_files/'

# Fit parameters
param_names = ['kappa', 'mu', 'alpha']

# Parameter limits
param_min = [0, 0.0005, 0.05]
param_max = [5e-4, 0.05, 10]

# Initial walker positions
param_start = [1e-4, 0.006, 4]

# Sampling
nwalkers = 24
nsteps = int(1e4)

# Prior distribution for mu
mu0 = 0.006
mustd = 0.0015

# DIUSST model
diffu_profile = 'LIN'
reflect = True
z_f = 10
dz = 0.10
ngrid = 40
sigma = 0.8
wind_max = 10
CFL = 0.95
ref_level = int(20)
humidity = 15

data_path = '../input_data/moce5/'
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
import os, sys
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
parent = os.path.dirname(dname)
os.chdir(dname)
sys.path.append(parent+'/src/')

# Load external modules
import numpy as np
import pandas as pd
import emcee
from multiprocessing import Pool, cpu_count

# Load custom functions
from diusst import Diusst

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
data_orig1 = pd.read_csv(data_path+data_filename)[data_interval1[0]:data_interval1[1]]

data_orig2_dirty = pd.read_csv(data_path+data_filename)[data_interval2[0]:data_interval2[1]]
data_orig2 = data_orig2_dirty.drop(data_orig2_dirty.index[removeidx2])

# Get mean foundation temperature
ftemp = np.mean(pd.read_csv(data_path+data_filename)['ftemp'].to_numpy())

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

# Adjust air temperature data to maintain correct air-sea temperature contrast after fixing foundation temperature
#data_orig1['atemp'] = data_orig1['atemp'] - data_orig1['ftemp'] + ftemp1
data_orig1['atemp'] = data_orig1['atemp'] - data_orig1['ftemp'] + ftemp
#data_orig2['atemp'] = data_orig2['atemp'] - data_orig2['ftemp'] + ftemp2
data_orig2['atemp'] = data_orig2['atemp'] - data_orig2['ftemp'] + ftemp
print('... Adjusted air temperature data relative to constant foundation temperature Tf = {:.2f}.'.format(ftemp))

# Set humidity to 15 g/kg
data_orig1['humid'] = data_orig1['humid']/10*humidity
data_orig2['humid'] = data_orig2['humid']/10*humidity
print('... Set humidity to {} g/kg.'.format(humidity))

###########################################################
# Define likelihood function
def bayesian_likelihood(params):
    kappa, mu, alpha = params

    model = Diusst(T_f=ftemp, kappa=kappa, mu=mu, alpha=alpha, sigma=sigma,
                    wind_max=wind_max, z_f=z_f, dz0=dz, ngrid=ngrid,
                    CFL=CFL, diffu_profile=diffu_profile, reflect=reflect)

    # interpolate to meet CFL condition
    data1, dtlist1, idx1 = model.interpolate(data_orig1, verbose=False)
    data2, dtlist2, idx2 = model.interpolate(data_orig2, verbose=False)

    simu1 = model.simulate(data1, progress=False)[0]
    simu2 = model.simulate(data2, progress=False)[0]

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
        log_prior = - (x[1]-mu0)**2/(2*mustd**2) - np.log( (param_max[0]-param_min[0])*(param_max[2]-param_min[2]) )
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

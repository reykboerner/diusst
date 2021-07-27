import numpy as np
import matplotlib.pyplot as plt
import h5py
import corner

import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

param_names = ['kappa','mu','attenu','kappa0','lambd']

name1 = '../output/210726-040109_M-I5-D1'

#parameter that excludes 0 in range
param_nonzero = 2
finish = True


file1 = h5py.File(name1+'.h5', 'r')
#file2 = h5py.File(name2+'.h5', 'r')
#file3 = h5py.File(name3+'.h5', 'r')

chain1 = np.array(file1.get('mcmc').get('chain'))
#chain2 = np.array(file2.get('mcmc').get('chain'))
#chain3 = np.array(file2.get('mcmc').get('chain'))
end = np.where(chain1[:,0,2]==0)[0][0]

print(chain1.shape)
print(chain1[end-1,:,:])

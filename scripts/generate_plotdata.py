__author__ = 'Reyk Boerner'

"""
Generates simulation data from the diuSST and slab models for plotting.
Parameters correspond to the MAP values estimated from Bayesian inference.
"""

# Import packages
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import sys
sys.path.append('../src/')
from atmosdata import AtmosData
from diusst import Diusst
from zengbeljaars import ZengBeljaars
from slab import Slab

# Load forcing data set
data = xr.load_dataset('../input_data/moce5/moce5_dataset.cdf', decode_timedelta=False)
time = data['time'].to_numpy()
dsst = data['dsst'].to_numpy()
wind = data['wind'].to_numpy()
T_f = float(data.T_f)

# Parameter values - diuSST
kappa, mu, alpha = 0.00013400773159919014, 0.002848626826335415, 3.518014095655488

# Parameter values - ZB05
nu, z_wind, C_drag, takaya10, skin_iter = 0.3, 10, 1.3e-3, True, 5

# Parameter values - slab
d, Q_sink, xi_1, xi_2 = 1.2010661501536875, 92.6740215866626, 0.00011854796494168126, 3.121690816078168e-11

# Instantiate models
model_diusst = Diusst(T_f=T_f, kappa=kappa, mu=mu, alpha=alpha, sigma=0.8, lambd=3, diffu_profile='LIN', CFL=0.95, z_f=10, dz0=0.1, ngrid=40)
model_zb05 = ZengBeljaars(T_f=T_f, z_wind=z_wind, nu=nu, C_drag=C_drag, takaya10=takaya10)
model_slab = Slab(d=d, Q_sink=Q_sink, xi_1=xi_1, xi_2=xi_2, T_f=T_f)
model_const = Slab(d=1e5, Q_sink=0, xi_1=1e-3, xi_2=0, T_f=T_f)

# Interpolate data in time
data_intp, dtlist, idx = model_diusst.interpolate(data)
atmosdata = AtmosData(data_intp)

# Run simulations
print('... Run diuSST model')
T_diusst, t_diusst, z_diusst, Q_diusst, terms = model_diusst.simulate(data_intp, output='detailed')
Qs_diusst, Ql_diusst, Rlw_diusst, Rsw_diusst = Q_diusst

print('... Run ZB05 model')
x_zb05, y_zb05, delta_zb05, Q_zb05 = model_zb05.simulate(atmosdata, [0,0,0.001], skin_iter=skin_iter, output="all")
Rlw_zb05, Qs_zb05, Ql_zb05 = Q_zb05
T_zb05 = x_zb05 + y_zb05

print('... Run slab model')
T_slab, Q_slab = model_slab.simulate(data_intp, output='detailed')
Qs_slab, Ql_slab, Rlw_slab = Q_slab

print('... Run constant SST simulation')
T_const, Q_const = model_const.simulate(data_intp, output='detailed')
Qs_const, Ql_const, Rlw_const = Q_const

# Remove interpolation points
T_diusst = T_diusst[idx].copy()
t_diusst = t_diusst[idx].copy()
Qs_diusst = Qs_diusst[idx].copy()
Ql_diusst = Ql_diusst[idx].copy()
Rlw_diusst = Rlw_diusst[idx].copy()

T_zb05 = T_zb05[idx].copy()
Qs_zb05 = Qs_zb05[idx].copy()
Ql_zb05 = Ql_zb05[idx].copy()
Rlw_zb05 = Rlw_zb05[idx].copy()
delta_zb05 = delta_zb05[idx].copy()

T_slab = T_slab[idx].copy()
Qs_slab = Qs_slab[idx].copy()
Ql_slab = Ql_slab[idx].copy()
Rlw_slab = Rlw_slab[idx].copy()

T_const = T_const[idx].copy()
Qs_const = Qs_const[idx].copy()
Ql_const = Ql_const[idx].copy()
Rlw_const = Rlw_const[idx].copy()

simu_diusst = T_diusst, t_diusst, z_diusst, [Qs_diusst, Ql_diusst, Rlw_diusst]
simu_zb05 = T_zb05, delta_zb05, [Qs_zb05, Ql_zb05, Rlw_zb05]
simu_slab = T_slab, [Qs_slab, Ql_slab, Rlw_slab]
simu_const = T_const, [Qs_const, Ql_const, Rlw_const]

# Store simulation data
np.savez('../output_files/simu_diusst.npz', *simu_diusst)
print('... saved simulation output as {}'.format('../../output_files/simu_diusst.npz'))
np.savez('../output_files/simu_zb05.npz', *simu_zb05)
print('... saved simulation output as {}'.format('../../output_files/simu_zb05.npz'))
np.savez('../output_files/simu_slab.npz', *simu_slab)
print('... saved simulation output as {}'.format('../../output_files/simu_slab.npz'))
np.savez('../output_files/simu_const.npz', *simu_const)
print('... saved simulation output as {}'.format('../../output_files/simu_const.npz'))

print('... Done.')

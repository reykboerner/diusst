# DiuSST, a diurnal sea surface temperature model
Simple 1D model of the diurnal warm layer, designed as an interactive oceanic boundary condition for cloud-resolving simulations.

> #### Quick links
> * [Preprint](https://arxiv.org/abs/2205.07933) (arXiv preprint)
> * [Starter tutorial](https://github.com/reykboerner/diusst/blob/master/docs/run_diusst.ipynb) (introductory example of running the code)
> * [Video presentation](https://youtu.be/KdOWF_fzRLE) (15-minute recorded talk)
>
>
> Contact: [reyk.boerner@reading.ac.uk](mailto:reyk.boerner@reading.ac.uk)


<p align = "center"><img src="https://github.com/reykboerner/diusst/blob/master/docs/header-image.png" alt="header-image" width="90%"/></p>

## About
This repository contains code for running and calibrating the *DiuSST* model. Have a look at the [summary](https://github.com/reykboerner/diusst/blob/master/docs/summary.md), [preprint](https://arxiv.org/abs/2205.07933) or [thesis](https://nbi.ku.dk/english/theses/masters-theses/reyk-borner/boerner_MSc_thesis.pdf) to learn more.

The main source code is located in the `src` folder. It contains Python implementations of the *DiuSST* model (`diusst.py`).
A Fortran implementation is in progress.

To exemplify how the model performs when forced with atmospheric data, the `input_data` folder contains an observational dataset from the MOCE-5 cruise in the Eastern Pacific. Furthermore, the `scripts` folder features code to calibrate the model parameters from data using Bayesian inference and MCMC sampling.

**Reproducability.** Results published in the [preprint](https://arxiv.org/abs/2205.07933) were produced with version `v1.1` of this repository. Code to reproduce figures is located in `scripts/figs`. Bayesian inference was performed using the scripts `paper_bayesian_diusst.py` (DiuSST model) and `paper_bayesian_slab.py` (slab model).

## Getting started

#### Prerequisites

The code is written in Python3 and requires the following modules:
* `numpy`, `scipy`, `xarray`, `tqdm`.

Additionally, for Bayesian inference we require
* `emcee`, `h5py`.

#### Running simulations with DiuSST

The *DiuSST* model is written as a Python class `Diusst`. The general procedure for running a simulation is
```
# load model
from diusst import Diusst

# create model instance
model = Diusst()

# interpolate atmospheric data set
data = model.interpolate(<Dataset>)

# run the model
simulation = model.simulate(data)
```

Model settings are specified through the `Diusst` class attributes. The interpolation step is required since the original forcing dataset may not have a sufficiently high temporal resolution for numerical stability.

Check out the [Jupyter notebook tutorial](https://github.com/reykboerner/diusst/blob/master/docs/run_diusst.ipynb) to learn how to run the code in practice and create a plot like the one above.

#### Calibrating model parameters

The `scripts` folder contains code to calibrate the model parameters based on a given data set (Bayesian inference). Using the `emcee` package, the code runs a Markov Chain Monte Carlo (MCMC) algorithm to sample the posterior distribution of parameters.

To run the script, change into the `scripts` folder and execute `python3 paper_bayesian_diusst.py` in a terminal.

Settings such as number of walkers and steps, location of the data set, and choice of prior can be modified directly in the `paper_bayesian_diusst.py` file.

Analogous to calibrating the parameters of *DiuSST*, the script `paper_bayesian_slab.py` performs Bayesian inference of the slab model parameters (see `slab.py`).

---

## Documentation

### The model
A detailed description of the model is provided in the [paper](https://arxiv.org/abs/2205.07933). The discretized model equation is given in eq. (A1).

In the Python script `src/diusst.py`, the model is written as a Python class with the following attributes:

#### Model parameters
| Label         | Parameter     | Units |
|--------------|-----------|------------|
| `T_f` | Foundation temperature $T_f$, constant (float) | K |
| `kappa` | Eddy diffusivity $\kappa_0$ (float, $\kappa >0$) | m²/s |
| `mu` | Mixing coefficient $\mu$ (float, $\mu>0$) | m/s |
| `alpha` | Attenuation coefficient $\alpha$ (float, $\alpha>0$) | 1/m |
| `sigma` | Surface suppressivity $\sigma$, used in EXP and STAB models (float, $0 \leq \sigma \leq 1$) |  |
| `lambd` | Trapping depth $\lambda$, used in EXP and STAB models (float, $\lambda>0$) | m |
| `z_ref` | Reference depth $z_r$, used in STAB model (float) | m |

#### Model options
| Label         | Description     | Values |
|--------------|-----------|------------|
| `diffu_profile` | Specifies the vertical diffusivity profile to be used (constant, linear, exponential, or stability-dependent) | `CONST` (default), `LIN` (default), `EXP`, `S-LIN`, `S-EXP` |
| `reflect` | Switch on/off reflection of downward shortwave radiation at the sea surface | bool (default `True`) |
| `CFL`| Target CFL number when determining the variable time step during data interpolation | float `CFL < 1` (default `0.95`) |
| `wind_max` | Cutoff maximum wind speed in the diffusion term, to limit computational cost (surface fluxes remain unaffected by this) | float (in units m/s, default `10` )|
| `wind_exp` | The exponent in the wind-dependence of turbulent diffusivity, i.e. $\kappa \sim u^w$, where $w$ is the `wind_exp`. In the thesis the dependence is quadratic, $w=2$ | float (default `2.0`) |



#### Model domain
| Label         | Description     | Units |
|--------------|-----------|------------|
| `z_f` | Foundation depth (float, default `10`) | m |
| `dz0` | Vertical grid spacing at surface (float, default `0.1`) | m |
| `ngrid` | Number of vertical grid points*. If `None`, then a uniform grid with spacing `dz0` is used (int, default `40`) |  |

(*) excluding the boundary grid points, i.e. the foundation point and atmospheric dummy point.

#### Constants
| Label         | Description     | Units | Value |
|--------------|-----------|------------|---|
| `k_mol` | Molecular diffusivity of sea water $\kappa_\text{mol}$ | m²/s | 1e-7 |
| `cp_w` | Specific heat of sea water (at const. pressure) | J/(kg K) | 3850 |
| `cp_a` | Specific heat of air (at const. pressure) | J/(kg K) | 1005 |
| `rho_w` | Density of sea water | kg/m³ | 1027 |
| `rho_a` | Density of air | kg/m³ | 1.1 |
| `n_w` | Refractive index of sea water | | 1.34 |
| `n_a` | Refractive index of air |  | 1.0 |
| `C_s` | Stanton number |  | 1.3e-3 |
| `C_l` | Dalton number |  | 1.5e-3 |
| `L_evap` | Latent heat of vaporization | J/kg | 2.5e6 |
| `sb_const` | Stefan-Boltzmann constant | W/(m K²)² | 5.67e-8 |
| `gas_const` | Gas constant of water vapor | J/(kg K) | 461.51 |


***

## Acknowledgements
This work has been conducted within the Atmospheric Complexity Group at the Niels Bohr Institute, University of Copenhagen, Denmark.

Collaborators: Romain Fiévet, Jan O. Haerter

We gratefully acknowledge Peter Minnett for providing meteorological and oceanographic data sets from the MOCE-5 cruise contained in this repository. The development and deployment of the instruments used during the cruise was funded by NASA.

I would further like to thank Peter Ditlevsen for co-supervising this project and Gorm G. Jensen for helpful discussions.

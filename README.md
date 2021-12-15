# DiuSST - Diurnal sea surface temperature model
A simple model of diurnal sea surface warming in the tropical ocean, designed as an interactive boundary condition for idealized atmospheric simulations.

> #### Quick links
> * [Thesis](https://github.com/reykboerner/diusst/blob/master/docs/boerner_MSc_thesis.pdf) (detailed description of the model in chapter 5, results in chapter 6)
> * [Presentation slides](https://github.com/reykboerner/diusst/blob/master/docs/boerner_MSc_defense.pdf) (from the thesis defense)
> * [Starter tutorial](https://github.com/reykboerner/diusst/blob/master/tutorials/run_diusst.ipynb) (introductory example of how to run the model in Python)
>
> Contact: [reyk.boerner@reading.ac.uk](mailto:reyk.boerner@reading.ac.uk)


<p align = "center"><img src="https://github.com/reykboerner/diusst/blob/master/docs/header-image.png" alt="header-image" width="90%"/></p>

## About
This repository contains code for running and analyzing the DiuSST model developed in the context of my master project. Have a look at the [summary](https://github.com/reykboerner/diusst/blob/master/docs/summary.md) or [thesis](https://github.com/reykboerner/diusst/blob/master/docs/boerner_MSc_thesis.pdf) to learn more. A paper about this work is in preparation and will be linked here when available.

Below, you will find information on how to use the code. (This repository is currently being expanded.)


## Getting started
The code essentially serves two purposes:
* run simulations with the DiuSST model to estimate the response of near-surface sea temperature to atmospheric forcing
* perform Bayesian inference to estimate model parameters given a dataset

#### Prerequisites
Clone the repo to your local hard drive using `git clone https://github.com/reykboerner/diusst.git`.

The code is written in Python3 and requires the following Python modules:
* numpy
* scipy
* pandas

Additionally, for Bayesian inference we require
* emcee
* h5py

Use `pip install <modulename>` to install the packages.

### Running simulations
The [Jupyter notebook tutorial](https://github.com/reykboerner/diusst/blob/master/tutorials/run_diusst.ipynb) provides an example of how to run the model and generate a plot like the one above. It uses a dataset from the MOCE-5 cruise, which is included in this repository.

As shown in the tutorial, running the DiuSST model corresponds to calling the `diusst` function. Model parameters and settings are adjusted through the arguments of this function. For a complete list of required and optional arguments, including their description and default values, click [here](#documentation).

### Bayesian inference

#### Modify settings
Settings of the MCMC run, parameter limits, the dataset to use etc. can be specified in the `run_bayesian.py` file found in `code`. The section where to edit these settings is labeled `# RUN SETTINGS` in the script.

#### Run
The script `run_bayesian.py` will read the specified data from the `data` folder and load `interpolation.py` to interpolate this data in time (ensuring that the CFL number meets the stability condition of the numerical model simulation).
To run the model, the script will load required functions from the `diusst_model.py` and `diusst_funcs.py` files.

Once you have checked the run settings, execute `python3 run_bayesian.py` to run the script.

#### Post-processing
While running, the script continuously writes the walker positions and log probabilities of the MCMC run into an `.h5` file labeled by a timestamp and run ID. This file can be used to recover output in case the run does not finish successfully.
After successful completion, the script stores the output in several files in the folder `output`.


## Documentation

### The model
A detailed description of the model is provided in chapter 5 of the [thesis](https://github.com/reykboerner/diusst/blob/master/docs/boerner_MSc_thesis.pdf).

In the Python script `diusst_model.py`, the model is written as a Python function `diusst` taking the following positional and keyword arguments:

#### *Positional arguments*
#### Input data: Timeseries of atmospheric variables
| Label         | Description     | Units |
|--------------|-----------|------------|
| `time_data` | Time values of atmospheric data timeseries (1D array)| s (since midnight local time) |
| `wind_data` | Wind speed data (1D array)| m/s |
| `swave_data` | Shortwave irradiance data (1D array)| W/m² |
| `airtemp_data` | Air temperature data (1D array)| K |
| `humid_data` | Air specific humidity data (1D array)| kg/kg |

#### *Keyword arguments*

#### Model parameters
| Label         | Parameter     | Units |
|--------------|-----------|------------|
| `T_f` | Foundation temperature $T_f$, constant (float) | K |
| `kappa` | Eddy diffusivity $\kappa_0$ (float, $\kappa >0$) | m²/s |
| `mu` | Mixing coefficient $\mu$ (float, $\mu>0$) | m/s |
| `alpha` | Attenuation coefficient $\alpha$ (float, $\alpha>0$) | 1/m |
| `lambd` | Trapping depth $\lambda$, used in EXP and STAB models (float, $\lambda>0$) | m |
| `sigma` | Surface suppressivity $\sigma$, used in EXP and STAB models (float, $0 \leq \sigma \leq 1$) |  |
| `z_ref` | Reference depth $z_r$, used in STAB model (float) | m |

#### Model options
| Label         | Description     | Values |
|--------------|-----------|------------|
| `diffu_type` | Specifies the vertical diffusivity profile to be used (constant, linear, exponential, or stability-dependent) | `BASE` (default), `LIN`, `EXP`, `STAB` |
| `init` | Initial condition for water temperature in the model domain. If `None`, then the foundation temperature `T_f` is used (float) | `None` (default) or float (in units K) |
| `output` | What data to output. Options are `temp` (water temperature array only), `basic` [water temperature, depth, time] or `detailed` [water temperature, depth, time, [sensible, latent, longwave, shortwave heat fluxes]] | `temp`, `basic` (default), `detailed` |
| `wind_max` | Cut-off wind speed for the turbulent diffusivity. Whenever `wind_data` $>$ `wind_max`, the value of `wind_max` will be used in the diffusivity term. (Surface fluxes remain unaffected by this.) (float)| float (in units m/s, default `10` )|
| `wind_exp` | The exponent in the wind-dependence of turbulent diffusivity, i.e. $\kappa \sim u^w$, where $w$ is the `wind_exp`. In the thesis the dependence is quadratic, $w=2$. (float) | float (default `2.0`) |
| `opac` | Opacity of the atmosphere to outgoing longwave radiation $\varepsilon_a$, $0\leq \varepsilon_a \leq 1$ | float (default `1.0`) |


#### Model domain
| Label         | Description     | Units |
|--------------|-----------|------------|
| `z_f` | Foundation depth (float, default `10`) | m |
| `dz` | Vertical grid spacing at surface (float, default `0.05`) | m |
| `ngrid` | Number of vertical grid points*. If `None`, then a uniform grid with spacing `dz` is used (int, default `None`) |  |

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

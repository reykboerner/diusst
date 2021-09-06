# DiuSST
An idealized model of diurnal sea surface temperature evolution, designed as an interactive boundary condition for idealized atmospheric simulations.

> Note: This repository is currently being reworked for enhanced user-friendliness and will soon feature additional descriptions.

## Quick start: Bayesian inference

### Prerequisites
Clone the repo to your local hard drive using `git clone https://github.com/reykboerner/diusst.git`.

The code runs in Python3 and requires the following python modules:
* numpy
* scipy
* pandas
* emcee
* h5py

Use `pip install <modulename>` to install the packages.

### Modify settings
Settings of the MCMC run, parameter limits, the dataset to use etc. can be specified in the `run_bayesian.py` file found in `code`. The section where to edit these settings is labeled `# RUN SETTINGS` in the script.

### Run
The script `run_bayesian.py` will read the specified data from the `data` folder and load `interpolation.py` to interpolate this data in time (ensuring that the CFL number meets the stability condition of the numerical model simulation).
To run the model, the script will load required functions from the `diusst_model.py` and `diusst_funcs.py` files.

Once you have checked the run settings, execute `python3 run_bayesian.py` to run the script.

### Post-processing
While running, the script continuously writes the walker positions and log probabilities of the MCMC run into an `.h5` file labeled by a timestamp and run ID. This file can be used to recover output in case the run does not finish successfully.
After successful completion, the script stores the output in several files in the folder `output`.

## Acknowledgements
This work has been conducted within the Atmospheric Complexity Group at the Niels Bohr Institute, University of Copenhagen, Denmark.

Collaborators: Romain Fiévet, Jan O. Haerter

We gratefully acknowledge Peter Minnett for providing meteorological and oceanographic data sets from the MOCE-5 cruise contained in this repository. The development and deployment of the instruments used during the cruise was funded by NASA.

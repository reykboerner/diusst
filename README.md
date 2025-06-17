# Conceptual models of the oceanic diurnal warm layer

[![](https://img.shields.io/badge/docs-dev-blue.svg)](#documentation) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13363481.svg)](https://doi.org/10.5281/zenodo.13363481)

This repository provides **three simple models** for simulating the **diurnal variability of sea surface temperature** (SST) under given atmospheric forcing:
- [**DiuSST**](https://doi.org/10.5194/gmd-18-1333-2025), a conceptual depth-resolved 1D model of upper ocean temperature dynamics (Fortran90, Python)
- The prognostic scheme of sea skin temperature by [**Zeng & Beljaars (2005)**](https://doi.org/10.1029/2005GL023030) (Python)
- A simple **slab ocean** with proportional and integral correctors (Python)

The repo also includes
- an **observational [dataset](#observational-dataset)** (MOCE-5 cruise, Eastern Pacific, 1999) for calibration and validation
- code to reproduce the results in the [model description paper](https://doi.org/10.5194/gmd-18-1333-2025) where all three models are compared with each other against observations.

Learn more about the DiuSST model in [this 15-minute video](https://youtu.be/KdOWF_fzRLE)!

<p align = "center"><img src="https://github.com/reykboerner/diusst/blob/master/docs/header-image.png" alt="header-image" width="90%"/></p>

## Model code

### Python
- For an **example notebook** to run DiuSST in Python, see `docs/run_diusst.ipynb`.
- This **DiuSST** code in `src/diusst.py` is documented [here](https://github.com/reykboerner/diusst/blob/master/docs/diusst.md).
- For information on running the **ZengBeljaars05** and **Slab** models, see the docstrings in `src/zengbeljaars.py` and `src/slab.py`.

### Fortran
The DiuSST model is also available as a Fortran90 code that can be coupled to atmospheric models (e.g. Large Eddy Simulations) in a few additional steps.

The source code is located in [`src/fortran/`](https://github.com/reykboerner/diusst/tree/master/src/fortran) containing the following files:
- `diusst.f90` provides a subroutine that evolves the SST field from one time step of the atmospheric model to the next.
- `forcing.f90` defines the surface forcing fields that are needed as inputs to run DiuSST.
- `parameters.90` specifies the model parameters.
- `grid.f90` specifies the horizontal and vertical domains.

This code is based on an implementation that couples DiuSST to the [System for Atmospheric Modeling](https://you.stonybrook.edu/somas/sam/) (SAM). In case of any questions, please contact [r.borner@uu.nl](mailto:r.borner@uu.nl).

## Observational dataset
The MOCE-5 cruise observations used to calibrate the DiuSST model as described in [the paper](https://doi.org/10.5194/gmd-18-1333-2025) is stored in `input_data/moce5/moce5_dataset.cdf` as a netCDF file. The raw data is also contained in the folder `input_data/moce5/`.

## Reproducibility
Results in [the paper](https://doi.org/10.5194/gmd-18-1333-2025) were produced with version `v1.2` of this repository. The script `scripts/generate_plotdata.py` runs the model simulations and saves the model output, which is found in `output_files` as `.npz` files. Code to reproduce figures based on these simulation data is located in `scripts/figs`. Model calibration via Bayesian inference was performed using the scripts `paper_bayesian_diusst.py` (DiuSST model) and `paper_bayesian_slab.py` (Slab model). The resulting posterior distributions are saved in `output_files` as `posterior_diusst.h5` and `posterior_slab.h5`, respectively.

## Acknowledgements
This work has been conducted within the Atmospheric Complexity Group at the Niels Bohr Institute, University of Copenhagen, Denmark.

Collaborators: Romain Fiévet, Jan O. Haerter

We gratefully acknowledge Peter Minnett for providing meteorological and oceanographic data sets from the MOCE-5 cruise contained in this repository. The development and deployment of the instruments used during the cruise was funded by NASA.

I would further like to thank Peter Ditlevsen for co-supervising this project and Gorm G. Jensen for helpful discussions. I am thankful to Chong Jia for a helpful discussion on the cool skin scheme in ZengBeljaars05.
# Conceptual models of the oceanic diurnal warm layer

[![](https://img.shields.io/badge/docs-dev-blue.svg)](#documentation) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.xx.svg)](https://doi.org/10.5281/zenodo.xx)

This repository contains Python implementations of three simple models for simulating the diurnal variability of sea surface temperature (SST) under given atmospheric forcing. It further includes an observational [dataset](#observational-dataset) for calibration and validation, plus code to reproduce the results in the [DiuSST model description paper](https://arxiv.org/abs/2205.07933).

Models in this repo:
* **DiuSST**, a conceptual depth-resolved 1D model of upper ocean temperature dynamics
* **ZengBeljaars05**, a prognostic scheme of sea skin temperature by Zeng & Beljaars (2005)
* **Slab**, a simple slab ocean with proportional and integral correctors

The *DiuSST* model is described in [arXiv:2205.07933](https://arxiv.org/abs/2205.07933), where it is compared to the *ZengBeljaars05* and *Slab* models based on observational data. The *ZengBeljaars05* scheme has originally been presented in [Zeng & Beljaars (2005)](https://doi.org/10.1029/2005GL023030). The *Slab* model is similar to responsive SST models used in idealized studies of tropical atmospheric convection, and is also described in [arXiv:2205.07933](https://arxiv.org/abs/2205.07933).

Learn more about the *DiuSST* model in [this 15-minute video](https://youtu.be/KdOWF_fzRLE)!

<p align = "center"><img src="https://github.com/reykboerner/diusst/blob/master/docs/header-image.png" alt="header-image" width="90%"/></p>

## Documentation
- The **DiuSST** model code is documented [here](https://github.com/reykboerner/diusst/blob/master/docs/diusst.md).
- For information on running the **ZengBeljaars05** and **Slab** models, see the docstrings in `src/zengbeljaars.py` and `src/slab.py`.
- An **example notebook** to run DiuSST is provided in `docs/run_diusst.ipynb`.

## Observational dataset
The MOCE-5 cruise observations used to calibrate the DiuSST model as described in [the paper](https://arxiv.org/abs/2205.07933) is stored in `input_data/moce5/moce5_dataset.cdf` as a netCDF file. The raw data is also contained in the folder `input_data/moce5/`.

## Reproducability
Results in the DiuSST model description [paper](https://arxiv.org/abs/2205.07933) were produced with version `v1.1` of this repository. The script `scripts/generate_plotdata.py` runs the model simulations and saves the model output, which is found in `output_files` as `.npz` files. Code to reproduce figures based on these simulation data is located in `scripts/figs`. Model calibration via Bayesian inference was performed using the scripts `paper_bayesian_diusst.py` (DiuSST model) and `paper_bayesian_slab.py` (Slab model). The resulting posterior distributions are saved in `output_files` as `posterior_diusst.h5` and `posterior_slab.h5`, respectively.

## Acknowledgements
This work has been conducted within the Atmospheric Complexity Group at the Niels Bohr Institute, University of Copenhagen, Denmark.

Collaborators: Romain Fiévet, Jan O. Haerter

We gratefully acknowledge Peter Minnett for providing meteorological and oceanographic data sets from the MOCE-5 cruise contained in this repository. The development and deployment of the instruments used during the cruise was funded by NASA.

I would further like to thank Peter Ditlevsen for co-supervising this project and Gorm G. Jensen for helpful discussions. I am thankful to Chong Jia for a helpful discussion on the cool skin scheme in ZengBeljaars05.
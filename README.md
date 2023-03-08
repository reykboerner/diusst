# Simple models of diurnal SST variability

This repository contains Python implementations of three simple models for simulating the diurnal variability of sea surface temperature (SST) under given atmospheric forcing, suitable as interactive boundary conditions for idealized cloud-resolving simulations.

* **diuSST**, a conceptual depth-resolved 1D model of upper ocean temperature dynamics
* **ZengBeljaars05**, a prognostic scheme of sea skin temperature by Zeng & Beljaars
* **Slab**, a simple slab ocean with proportional and integral correctors

The *diuSST* model is described in [Börner et al. (2022)](https://arxiv.org/abs/2205.07933), where it is compared to the *ZengBeljaars05* and *Slab* models based on observational data. The *ZengBeljaars05* scheme has originally been presented in [Zeng & Beljaars (2005)](https://doi.org/10.1029/2005GL023030). The *Slab* model is similar to responsive SST models used in idealized studies of tropical atmospheric convection.

Learn more about the *diuSST* model in [this 15-minute video](https://youtu.be/KdOWF_fzRLE)!

<p align = "center"><img src="https://github.com/reykboerner/diusst/blob/master/docs/header-image.png" alt="header-image" width="90%"/></p>

## Getting started

For more information, see the documentation for [diuSST](https://github.com/reykboerner/diusst/blob/master/docs/diusst.md), [ZengBeljaars05](), and [Slab]().

## Acknowledgements
This work has been conducted within the Atmospheric Complexity Group at the Niels Bohr Institute, University of Copenhagen, Denmark.

Collaborators: Romain Fiévet, Jan O. Haerter

We gratefully acknowledge Peter Minnett for providing meteorological and oceanographic data sets from the MOCE-5 cruise contained in this repository. The development and deployment of the instruments used during the cruise was funded by NASA.

I would further like to thank Peter Ditlevsen for co-supervising this project and Gorm G. Jensen for helpful discussions. I am thankful to Chong Jia for a helpful discussion on the cool skin scheme in ZengBeljaars05.
# BayesianPMs
Use the outputs from [GaiaHub](https://github.com/AndresdPM/GaiaHub/) (see also [del Pino et al. 2022](https://ui.adsabs.harvard.edu/abs/2022ApJ...933...76D/abstract/)) to measure proper motions, parallaxes, and positions in a heirarchical Bayesian framework from Gaia and archival Hubble images.

Works best for sources in the 17 < G < 21.5 mag range in medium- to low-density environemnts. High-density environments are likely to have incorrect cross-matching between Gaia and Hubble sources. 

The Gaia-measured values are used as priors where they exist. This pipeline uses an MH-MCMC approach to measure 6 transformation parameters (scale, rotation, on- and off-skews, and center position) between the Gaia pseudo-image and the HST image while concurrently measuring posterior distributions on proper motion, parallax, and position for every source. Multiple images can be analysed together, and the statistics/math is general in that it applies to any two or more images. 


Here is an example call, assuming that an output directory from GaiaHub is named "Fornax_dSph":
```
python GaiaHub_BPPPM.py --name "Fornax_dSph"
```

# faraday

This code performs the analysis described by Oppermann et al. 2015 (2015A&amp;A...575A.118O).
The results of the original analysis can be found at http://wwwmpa.mpa-garching.mpg.de/ift/faraday/


## Dependencies

* numpy (http://www.numpy.org)
* scipy (https://www.scipy.org)
* healpy (https://healpy.readthedocs.io)
* nifty (http://wwwmpa.mpa-garching.mpg.de/ift/nifty/)


## Reconstruction procedure

The code follows the model of 2015A&amp;A...575A.118O, i.e., a split of observed Faraday rotation data into the sum of a Galactic component, an extragalactic component, and noise.

The Galactic component is modeled as the product of a dimensionless foreground component, assumed to be an isotropic Gaussian random field with an unknown angular power spectrum, and a profile function that depends only on Galactic latitude.

The data are split into two categories (within the `read_data` routine), one that is afflicted by uncertainties about the error bars (SIP data in the paper) and one that is not (VIP data in the paper).

To reconstruct this split, the code iteratively calculates new estimates for
* the Galactic Faraday depth
* the extragalactic contribution to the sources' Faraday depth
* correction factors for the error variances of the SIP points
* a correction factor for the typical extragalactic contribution (initially assumed to be 6.6 rad/m^2)
* the angular power spectrum of the dimensionless foreground component
* the profile function that relates the dimensionless foreground component to the physical Galactic Faraday depth

The profile function is updated only once every 10 iterations (in the current implementation), since it involves calculating an estimate for the uncertainty of the foreground reconstruction, which is done via operator probing and takes quite long. This update is referred to in the code as a toplevel iteration. Each toplevel iteration contains 10 successive updates of all other quantities.

Operators used in the code are implemented as operations using the nifty operator framework, i.e., matrices aren't stored explicitly. When diagonals or traces of operators are calculated, this is done via operator probing (see, e.g., 
2012PhRvE..85b1134S or the nifty documentation). To this end, a number of CPUs can be specified in the code to be used to spawn parallel processes. To make this work most efficiently, this number should be large and Open MP parallelization of any involved routines should be turned off (most easily by setting the environment variable OMP_NUM_THREADS to 1).


## Usage

The code unelegantly consists of a single python script that can be run from the command line by invoking
  
    python reconstruct.py

Before that, however, you will have to make a few adjustments:

* *Data file*: Currently, the data are read in using the `read_data` routine, which assumes that all the data are in one file, currently assumed to be named `completedata.txt` and consisting of four columns:
  1. Galactic longitude in radians
  2. Galactic colatitude in radians (the colatitude is the usual theta coordinate in spherical polar coordinates, which is zero at the North pole)
  3. observed Rotation measure value in rad/m^2
  4. one-sigma error bar on the RM value in rad/m^2
  
  The code presently also has a lookup table that assigns authors to each data point. It corrects the error bars of the NVSS RM catalog (Taylor et al. 2009; 2009ApJ...702.1230T) by a factor 1.22 following Stil et al. (2011ApJ...726....4S) and assigns the data points to the two categories (SIP and VIP) according to this lookup table. Currently, only the data of Mao et al. 2010 (2010ApJ...714.1170M) are regarded as VIP.
  
  All these things will have to be updated when using other data or wanting to make other choices.

* *CPU usage*: In the first section of the code, a number of CPUs to be used can be specified. This is relevant for the operator probing (whenever diagonals or traces are calculated). Don't choose a number that's larger than the available number of CPUs and don't forget to set the environment variable `OMP_NUM_THREADS` to 1 before running the code.

* *Resolution*: In the first section of the code, the resolution of the desired foreground map can be specified via the HEALPix `Nside` parameter (see http://healpix.jpl.nasa.gov). When using a much lower resolution, the smoothing scale in the calculation of the profile function, currently hard-coded to 4 degrees, will have to be increased in the routines `calc_profilema` and `calc_profilemap_from_map` to avoid numerical issues occurring when smoothing with a Gaussian that's much smaller than a pixel.

* *Other parameters*: In the first section of the code, there are a few other parameters that one may want to adapt:
  * strength of the spectral smoothness prior (`var_p`)
  * parameters of the prior for the angular power spectrum (`q` and `alpha`)
  * parameter for the prior on the error variance correction factors (`beta`)
  * The number of data points and number of VIP data points (`ndata` and `ndata_VIP`) are currently provided by hand, mostly for simplicity and as a cross-check that nothing is going wrong when reading in the data.
  * A directory for the output needs to be specified (`rundir`).

* *Convergence*: Presently, all iterations are implemented as for-loops. This means that convergence needs to be determined by eye and more or fewer iterations should be run if desirable. The code can easily be restarted after a toplevel iteration.


### Output

The following files should appear in the output directory as the iteration progresses:
* `profmap_XX.npy`: profile function in map form (for ease of use; in rad/m^2; only calculated once per toplevel iteration)
* `prof_XX`: profile function as an ASCII file; columns are Galactic latitude (in degrees) and profile function (in rad/m^2; only calculated once per toplevel iteration)
* `Cl_XX`: angular power spectrum as an ASCII file; rows are l = 0,1,2,....
* `m_g_XX.npy`: map of the dimensionless foreground component
* `m_g2_XX.npy`: map of the dimensionless foreground component made using only the VIP data (as an intermediate step)
* `m_e_XX.npy`: estimate for the extragalactic contribution to all data points (in rad/m^2)
* `eta_i_XX`: correction factors for the error variances for the SIP data (according to Eq. 29 in the paper, where sigma_e and sigma_i are the initial guesses)
* `eta_e_XX`: correction factor for the extragalactic variance
* `Dhat_XX.npy`: map with an estimate for the uncertainty variance of the dimensionless foreground component (only calculated once per toplevel iteration)

All `.npy` files are binary numpy files, all others are ASCII files.

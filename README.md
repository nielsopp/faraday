# faraday

This code performs the analysis described by Oppermann et al. 2015 (2015A&amp;A...575A.118O).
The results of the original analysis can be found at http://wwwmpa.mpa-garching.mpg.de/ift/faraday/


# Dependencies

* numpy (http://www.numpy.org)
* scipy (https://www.scipy.org)
* healpy (https://healpy.readthedocs.io)
* nifty (http://wwwmpa.mpa-garching.mpg.de/ift/nifty/)


# Reconstruction procedure

The code follows the model of 2015A&amp;A...575A.118O, i.e., a split of observed Faraday rotation data into the sum of a Galactic component, an extragalactic component, and noise.

The Galactic component is modeled as the product of a dimensionless foreground component, assumed to be an isotropic Gaussian random field with an unknown angular power spectrum, and a profile function that depends only on Galactic latitude.

The data are split into two categories (within the 'read_data' routine), one that is afflicted by uncertainties about the error bars (SIP data in the paper) and one that is not (VIP data in the paper).

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

# -*- coding: utf-8 -*-
"""
This is code that should be able to redo the analysis of 2015A&A...575A.118O.

@author: niels
"""


from nifty import *
from scipy.sparse.linalg import LinearOperator as lo
from scipy.sparse.linalg import cg
import sys
import scipy.integrate as integr
import math
#import pyfits as pf
import astropy.io.fits as pf

note = notification()
about.lm2gl.off()


#==============================================================================
# Parameters of the reconstruction
#==============================================================================


#random seed:
#np.random.seed(101)

#number of CPUs to use for operator probing (remember to set
#OMP_NUM_THREADS=1 when using CPU > 1):
CPU = 4

#resolution of the map (HEALPix NSIDE parameter):
nside = 128

#variance associated with the spectral smoothness prior for the power spectrum
#of the Galactic component (large number means no smoothness enforced)
var_p = 10.

#parameters describing the inverse-Gamma prior for the Galactic power spectrum
#(Eq. 24), assumed to be scale-independent:
q = 0.
alpha = 1.

#parameter describing the inverse-Gamma prior for the eta-factors (Eq. 30),
#assumed to be constant:
beta = 2.

#number of data points (for consistency check):
ndata = 41632

#number of VIP data points:
ndata_VIP = 729

#name of a directory in which to save the output (absolute or relative):
rundir = 'test/'


#==============================================================================
# Setting up spaces
#==============================================================================


px = hp_space(nside)
lm = px.get_codomain()
ds = point_space(ndata)
ds2 = point_space(ndata_VIP)

llengths,rho,ls,pundex = lm.get_power_indices()
lmax = ls.max()

fm = ncmap.fm()
fu = ncmap.fu()
fm.set_under([0,0,0,0])
fu.set_under([0,0,0,0])


#==============================================================================
# Class definitions
#==============================================================================


class propagator_operator(operator):
    """
        Class for the operator D used in the paper.
        The way this is written is unnecessarily complicated for historical
        reasons.
        The inverse of D is implemented and a conjugate gradient routine is
        used to invert that.
    """

    def __init__(self,domain,Cl,Mlmdiag,M,acc=0.05,imp=True):
        if(not isinstance(domain,space)):
            raise TypeError("ERROR: invalid input.")
        self.domain = domain
        self.imp = bool(imp)
        self.sym = True
        self.uni = False
        self.target = self.domain
        self.Cl = Cl
        self.Mlmdiag = Mlmdiag
        self.M = M
        self.acc = acc
        Pdiag = 1.
        self.P = diagonal_operator(self.domain.get_codomain(),diag=Pdiag)
        PSinvdiag = 1./Cl[ls]
        self.PSinv = diagonal_operator(self.domain.get_codomain(),
                                       diag=PSinvdiag)

    def _inverse_multiply(self,x):
        return self.PSinv(x) + self.P(self.M(x))
    
    _matvec = (lambda self,x: self.inverse_times(x).val.flatten())

    def _multiply(self,x):
        A = lo(shape=tuple(self.dim()),matvec=self._matvec,
               dtype=self.domain.datatype)
        b = self.P(x).val.flatten()*1.e10

        def callback(xk):
            print ((((self.inverse_times(xk) - b))**2).val.sum()/(b**2).sum())**0.5
            sys.stdout.flush()
            return 0

        x_,info = cg(A,b,x0=None,tol=self.acc,callback=None)
        
        if(info==0):
            if np.all(x_==0.):
                print 'CG did not run.'
            return x_*1.e-10
        else:
            note.cprint("NOTE: conjugate gradient failed.")
            return None


class RDRdagger_operator(operator):
    """
        Operator class for R D R^dagger; needed, e.g., in Eq. (B.15).
    """
    
    def _multiply(self,x):
        R = self.para[0]
        D = self.para[1]
        return R(D(R.adjoint_times(x)))


class aux_operator(operator):
    """
        Auxiliary operator class used to implement Eq. (B.25).
    """
    def _multiply(self,x):
        R = self.para[0]
        D2 = self.para[1]
        R2 = self.para[2]
        NpE2 = self.para[3]
        out = R2(D2(R.adjoint_times(x)))
        out = NpE2.inverse_times(NpE2.inverse_times(out))
        out = R(D2(R2.adjoint_times(out)))
        return out


#==============================================================================
# Helper functions
#==============================================================================


def findauthor(i):
    """
        Function to find out which catalog a data point is from, according to
        where it appears in my data file; figured this out by hand once. Not
        elegant and obviously only working when using exactly this file.
    """
    if i<7:
        return 'Bonafede'
    elif i<64:
        return 'Heald'
    elif i<345:
        return 'Feain'
    elif i<533:
        return 'Mao LMC'
    elif i<595:
        return 'Mao SMC'
    elif i<995:
        return 'Mao NorthCap'
    elif i<1324:
        return 'Mao SouthCap'
    elif i<1392:
        return 'Johnston-Hollitt A'
    elif i<1570:
        return 'Schnitzeler'
    elif i<1718:
        return 'Brown SGPS'
    elif i<1764:
        return 'O\'Sullivan'
    elif i<39307:
        return 'Taylor'
    elif i<39432:
        return 'Clarke'
    elif i<39520:
        return 'Hammond'
    elif i<39587:
        return 'Roy'
    elif i<39781:
        return 'Van Eck'
    elif i<39793:
        return 'Johnston-Hollitt B'
    elif i<39936:
        return 'Klein'
    elif i<39992:
        return 'Clegg'
    elif i<40090:
        return 'Minter'
    elif i<40108:
        return 'Gaensler'
    elif i<40488:
        return 'Brown CGPS'
    elif i<41023:
        return 'Simard-Normandin'
    elif i<41024:
        return 'Simard-Normandin / Rudnick'
    elif i<41025:
        return 'Simard-Normandin / Tabara'
    elif i<41027:
        return 'Simard-Normandin / Oren'
    elif i<41028:
        return 'Simard-Normandin / Broten'
    elif i<41045:
        return 'Rudnick'
    elif i<41046:
        return 'Rudnick / Oren'
    elif i<41051:
        return 'Wrobel'
    elif i<41052:
        return 'Wrobel / Kim'
    elif i<41114:
        return 'Tabara'
    elif i<41115:
        return 'Tabara / Oren'
    elif i<41116:
        return 'Tabara / Broten'
    elif i<41167:
        return 'Oren'
    elif i<41288:
        return 'Broten'
    elif i<41289:
        return 'Broten / Simard-Normandin'
    elif i<41309:
        return 'Kim'
    elif i<41312:
        return 'Lawler'
    elif i<41329:
        return 'Hennessy'
    elif i<41330:
        return 'Kato'
    elif i<41632:
        return 'Mao 2012'
    else:
        raise NameError('Index out of dataset.')


def read_data():
    """
        Function that reads in the data and assigns the categories
        (0 for SIP and 1 for VIP).
        Here it is assumed that Mao's data are VIP, all others are SIP.
        Will only work if using my data file.
    """
    stuffold = np.genfromtxt('completedata.txt')
    lon = stuffold[:,0]*180./np.pi
    lat = 90. - stuffold[:,1]*180./np.pi
    d = stuffold[:,2]
    sigma = stuffold[:,3]
    category = np.zeros(len(stuffold),dtype=bool)
    for i in range(len(stuffold)):
        if findauthor(i) == 'Taylor':
            sigma[i] *= 1.22
        if findauthor(i) in ['Mao SouthCap', 'Mao NorthCap']:
            category[i] = 1
    return lon, lat, d, sigma, category


def calc_pixels(lon,lat):
    """
        Figure out which HEALPix pixels the data lie in.
    """
    pixels = hp.ang2pix(nside,np.pi/2. - lat/180.*np.pi,lon/180.*np.pi)
    return pixels


def calc_profmap_from_prof(prof):
    """
        Helper routine that turns a profile function into a full HEALpix map.
    """
    profmap = field(px)
    pix = 0
    for i in range(nside):
        for j in range(4*(1 + i)):
            profmap[pix] = prof[i]
            pix += 1
    for i in range(nside,3*nside - 1):
        for j in range(4*nside):
            profmap[pix] = prof[i]
            pix += 1
    for i in range(3*nside - 1,4*nside - 1):
        for j in range(4*(4*nside - 1 - i)):
            profmap[pix] = prof[i]
            pix += 1
    if pix != px.dim():
        raise NameError('Error in the calculation of the profile map.')
    return profmap

    
def calc_profile_from_profilemap(profmap):
    """
        Helper routine that turns a HEALpix map into a profile function.
    """
    nring = 4*nside - 1
    pix = 0
    profile = np.zeros(nring)
    for i in range(nside):
        profile[i] = profmap[pix]
        pix += 4*(1 + i)
    for i in range(nside,3*nside - 1):
        profile[i] = profmap[pix]
        pix += 4*nside
    for i in range(3*nside - 1,nring):
        profile[i] = profmap[pix]
        pix += 4*(4*nside - 1 - i)
    if pix != px.dim():
        raise NameError('Error inthe calculation of the profile.')
    latitudes = 90. - 90./nring - np.arange(nring)*180./nring
    return latitudes,profile


def calc_profilemap(pixels,d,sigma,lat,lon):
    """
        Estimating a profile map from a raw data set.
        A smoothing with sigma = 4 degrees is performed.
    """
    nring = 4*nside - 1
    binsize = 180./nring
    rms = np.zeros(nring)
    for i in range(nring):
        a = (d[(lat < 90. - binsize*i)
               & (lat >= 90. - binsize*(i + 1))]**2 - sigma[(lat < 90. - binsize*i) & (lat >= 90. - binsize*(i + 1))]**2).mean()
        rms[i] = a**0.5
    if np.isnan(rms[0]):
        k = 1
        while np.isnan(rms[k]):
            k += 1
        rms[0] = rms[k]
    if np.isnan(rms[nring - 1]):
        k = nring - 2
        while np.isnan(rms[k]):
            k -= 1
        rms[nring - 1] = rms[k]
    for i in range(1,nring - 1):
        if np.isnan(rms[i]):
            k = i - 1
            while np.isnan(rms[k]):
                k -= 1
            l = i + 1
            while np.isnan(rms[l]):
                l += 1
            rms[i] = ((rms[k]**2/(i - k) + rms[i - 1]**2/(l - i))/(1./(i - k) + 1./(l - i)))**0.5
    profmap = calc_profmap_from_prof(rms)
    return profmap.smooth(sigma=4./180.*np.pi)


def calc_profilemap_from_maps(phi,Dphihat):
    """
        Estimating the profile map from a reconstruction of the Galactic
        component and its uncertainty.
        A smoothing with sigma = 4 degrees is performed.
    """
    nring = 4*nside - 1
    squaremap = hp.smoothing(phi**2 + Dphihat,4./180.*np.pi)
    expectedsquares = np.zeros(nring)
    pix = 0
    for i in range(nside):
        for j in range(4*(1 + i)):
            expectedsquares[i] += squaremap[pix]
            pix += 1
        expectedsquares[i] /= 4*(1 + i)
    for i in range(nside,3*nside - 1):
        for j in range(4*nside):
            expectedsquares[i] += squaremap[pix]
            pix += 1
        expectedsquares[i] /= 4*nside
    for i in range(3*nside - 1,nring):
        for j in range(4*(4*nside - 1 - i)):
            expectedsquares[i] += squaremap[pix]
            pix += 1
        expectedsquares[i] /= 4*(4*nside - 1 - i)
    if pix != px.dim():
        raise NameError('Error in the calculation of the profile map.')
    profmap = calc_profmap_from_prof(expectedsquares**0.5)
    return profmap


#==============================================================================
# critical filter functions
#==============================================================================


def calc_Laplace(ls_irr_binned):
    """
        Calculates the discretized version of the Laplacian operator according
        to Eqs. (B5) and (B6) of Oppermann et al. 2013. and a diagonal matrix
        containing the volume factors for a continuous scalar product in the
        power spectrum space.
        
        Parameters
        ----------
        lmax : int
            Maximum index for the power spectrum (default: None).
        ls_irr_binned : numpy.ndarray
            Array containing the physical lengths (irreducible power indices)
            of the power spectrum modes. If binning is used, use the lengths
            corresponding to the bins.
        
        Returns
        -------
        matrix : numpy.ndarray
            The discretized Laplace operator.
        integr : numpy.ndarray
            The diagonal matrix needed for the implementation of continuous
            scalar products in the power spectrum space.
    """
    matrix = np.zeros((lmax + 1,lmax + 1))
    ltilde = np.log(ls_irr_binned/ls_irr_binned[1])
    for i in range(2,lmax):
        matrix[i,i-1] = 2./(ltilde[i+1] - ltilde[i-1])/(ltilde[i] - ltilde[i-1])
        matrix[i,i] = -2./(ltilde[i+1] - ltilde[i-1])*(1./(ltilde[i+1] - ltilde[i]) + 1./(ltilde[i] - ltilde[i-1]))
        matrix[i,i+1] = 2./(ltilde[i+1] - ltilde[i-1])/(ltilde[i+1] - ltilde[i])
    
    integrand = np.zeros((lmax + 1,lmax+1))
    for i in range(2,lmax):
        integrand[i,i] = (ltilde[i+1] - ltilde[i-1])/2.
    return matrix, integrand


def calc_inv_matrix(p,var_p_here):
    """
        Calculating the inverse of the Hessian matrix used in the
        Newton-Raphson method in the routine `calc_power`, as well as the term
        rho + 2*Tp.
        
        Parameters
        ----------
        lmax : int
            Maximum index for the power spectrum (default: None).
        p : numpy.ndarray
            Natural logarithm of the power spectrum.
        var_p : float
            Variance used in the spectral smoothness prior, Eq. (25), i.e.
            sigma_p^2.
        
        Returns
        -------
        inv_matrix : numpy.ndarray
            Inverse Hessian used in the Newton-Raphson method in the routine
            `calc_power`.
        denom : numpy.ndarray
            Other term, 2*Tp + rho, used in the routine `calc_power`.
        
        Notes
        -----
        The quantity `T` as defined in the code corresponds to 2*T as defined
        in Oppermann et al. 2013.
    """
    laplace, integr = calc_Laplace(llengths)
    laplacedagger = np.transpose(laplace)
    T = np.dot(integr,laplace)
    T = np.dot(laplacedagger,T)*2./var_p_here    # Eq. (B8)*2
    Tp = np.dot(T,p)
    for i in range(lmax + 1):
        T[i,i] += Tp[i] + rho[i]
    denom = rho + Tp
    return np.linalg.inv(T), denom


def calc_power(D,m):
    """
        Calculates a new guess for the power spectrum by solving Eq. (B.12) for
        C_l. This is done using a Newton-Raphson method with adaptive stepsize
        and a temporary increase of the variance of the spectral smoothness
        prior.
        
        Parameters
        ----------
        D : nifty.operator
            Current estimate for the propagator.
        m : nifty.field
            Current estimate for the map.
        var_p : float
            Variance used in the spectral smoothness prior, Eq. (24), i.e.
            sigma_p^2.
        
        Returns
        -------
        Cl : numpy.ndarray
            New estimate for the power spectrum.
        
        Notes
        -----
        The parameter alpha of the inverse-Gamma prior is assumed to be one.
        The quantity `T` as defined in the code corresponds to 2*T as defined
        in Oppermann et al. 2013.
    """
    mdaggerm = m.transform().val*m.transform().val.conjugate()
    diag_D_k = D.diag(domain=lm,bare=True,ncpu=CPU,nrun=10,loop=True)
    delta = 1.
    B = (mdaggerm + delta*diag_D_k)*lm.get_meta_volume()
    Clnum = (2.*q + np.array([np.sum(B[np.where(ls==k)[0]]).real
                              for k in range(lmax + 1)]))/np.prod(lm.vol)
    Cl = Clnum/rho
    delta = np.ones(lmax + 1)   # increment for the logarithmic power spectrum
    var_p_here = var_p*1.1  # temporally increading the variance
    breakinfo = 0
    print 'smoothing started'
    while var_p_here >= var_p: # slowly lowering the variance
        absdelta = 1.
        while absdelta > 1.e-3: # solving with fixed variance
            inv_matrix, denom = calc_inv_matrix(np.log(Cl),var_p_here)
            delta = np.dot(inv_matrix,Clnum/Cl - denom)
            if np.abs(delta).max() > absdelta:  # increasing variance when speeding up
                var_p_here *= 1.1
            absdelta = np.abs(delta).max()
            fraction = np.min([0.1/absdelta,1.0])   # adaptive stepwidth
            p = np.log(Cl) + fraction*delta
            Cl = np.exp(p)
        var_p_here /= 1.1   # lowering the variance when converged
        if var_p_here < var_p:
            if breakinfo:   # making sure there's one iteration with the correct variance
                break
            var_p_here = var_p
            breakinfo = 1
    print 'smoothing finished'

    Cl[Cl<1.e-30] = 1.e-30  # Some ad-hoc fixes, which may or may not be needed.
    Cl[0] = Cl[1]
    return Cl


#==============================================================================
# functions for the extended critical filter
#==============================================================================


def calc_r(beta):
    """
        Calculating the parameter r from the paramter beta in the prior for the
        noise variance correction factors, such that the prior expectation
        value of their logarithm becomes 0.
    """
    f = lambda x : np.log(x)*x**(beta-2)*np.exp(-x)
    a = integr.quad(f,0.,np.infty)[0]
    r = np.exp(a/math.factorial(beta-1))
    return r


def calc_eta(d,R,R2,m,m2,RDRdagger,RD2Rdagger,D2,eta_i,eta_e,sigma,sigma_e,
             category,aux,do_eta_e=True):
    """
        Calculating the correction factors for the error bars and for the
        extragalactic variance according to Eqs. (B.15) and (B.23).
    """
    diffsq = (d - R(m))**2
    if do_eta_e:
        diffsq2 = (d - R(m2))**2
    RDRdaggerdiag = RDRdagger.diag(bare=True,nrun=100,ncpu=CPU)
    print 'RDRdaggerdiag done.'
    if do_eta_e:
        RD2Rdaggerdiag = RD2Rdagger.diag(bare=True,nrun=10,ncpu=CPU)
        print 'RD2Rdaggerdiag done.'
        auxdiag = aux.diag(bare=True,nrun=10,ncpu=CPU)
        print 'auxdiag done'

    crit = (eta_e*sigma_e**2 + sigma[category==0]**2)/(sigma[category==0]**2 + sigma_e**2)
    r = 1.5*np.maximum(1.,crit)

    temp = 0.5*(diffsq + np.maximum(0.,RDRdaggerdiag))[category==0]
    temp /= sigma[category==0]**2 + sigma_e**2
    temp += r
    eta_i = temp/1.5

    if do_eta_e:
        temp = (diffsq2 + RD2Rdaggerdiag)[category==1]
        temp *= eta_e**2*sigma_e**2/(eta_e*sigma_e**2 + sigma[category==1]**2)**2
        out = 0.5*temp.sum()
        denom = eta_e*sigma_e**2/(eta_e*sigma_e**2 + sigma[category==1]**2)
        denom = 0.5*denom.sum()
        corr = (d[category==1] - R2(m2))/(eta_e*sigma_e**2 + sigma[category==1]**2)**2
        corr = R(D2(R2.adjoint_times(corr)))
        corr = (d - R(m2))*corr
        corr *= 2.
        corr -= auxdiag
        corr = corr[category==0]
        corr /= r*(sigma[category==0]**2 + sigma_e**2) + 0.5*(diffsq2 + RD2Rdaggerdiag)[category==0]
        corr = 1.5*corr.sum()
        print 'eta_e =', out/denom
        print 'correction =', corr/denom
        
        eta_e = out/denom + corr/denom
    
        if eta_e <= 0.:
            print 'WARNING: eta_e has gone negative.'
            eta_e *= (-1)

    return eta_i, eta_e


#==============================================================================
# Main routine
#==============================================================================


def reconstruct():

    # reading in data and creating auxiliary objects
    lon, lat, d, sigma, category = read_data()
    ncat1 = (category==0).sum()
    ncat2 = (category==1).sum()
    pixels = calc_pixels(lon,lat)
    profmap = calc_profilemap(pixels,d,sigma,lat,lon)
    np.save(rundir + 'profmap00.npy',profmap)
    lat, prof = calc_profile_from_profilemap(profmap)
    np.savetxt(rundir + 'prof_00',np.array([lat,prof]).transpose())
    print 'number of good data points:', ncat2
    print 'number of not-so-good data points:', ncat1
        
    # initial power spectrum
    Cl = 1.53*np.arange(lmax + 1)**(-2.17)
    Cl[0] = Cl[1]
    np.savetxt(rundir + 'Cl_%02u'%0,Cl)
    eta_i = np.ones(ncat1)
    eta_e = 1.

    sigma_e = 6.6

    #the toplevel iteration corresponds to successive estimates for the profile
    #function (adapt the number of iterations manually to make sure everything
    #has converged):
    for toplevel in range(10):

        if toplevel > 0:
            #The reconstruction should be easy to restart at any toplevel
            #iteration. Here, the results from the previous toplevel iteration
            #are loaded:
            m_g = np.load(rundir + 'm_g_%02i.npy'%(toplevel*10))
            profmapold = np.load(rundir + 'profmap%02i.npy'%((toplevel-1)*10))
            Dhat = np.load(rundir + 'Dhat_%02i.npy'%(toplevel*10))
            phi = profmapold*m_g
            Dhatphi = profmapold**2*Dhat
            
            #calculate a new estimate for the profile map:
            profmap = calc_profilemap_from_maps(phi,Dhatphi)
            np.save(rundir + 'profmap%02i.npy'%(toplevel*10),profmap)
            lat, prof = calc_profile_from_profilemap(profmap)
            np.savetxt(rundir + 'prof_%02i'%(toplevel*10),np.array([lat,prof]).transpose())

            #load some more old results:
            Cl = np.genfromtxt(rundir + 'Cl_%02i'%(toplevel*10))
            eta_i = np.genfromtxt(rundir + 'eta_i_%02i'%(toplevel*10))
            eta_e = np.genfromtxt(rundir + 'eta_e_%02i'%(toplevel*10))

        # setting up operators
        NpE = diagonal_operator(ds,diag=sigma**2 + sigma_e**2,bare=True)
        newdiag = sigma**2
        newdiag[category==0] += sigma_e**2
        newdiag[category==0] *= eta_i
        newdiag[category==1] += eta_e*sigma_e**2
        NpE.set_diag(newdiag,bare=True)
        NpE2 = diagonal_operator(ds2,diag=sigma[category==1]**2 + eta_e*sigma_e**2,bare=True)
        R = response_operator(px,target=ds,mask=profmap,assign=pixels,den=False)
        R2 = response_operator(px,target=ds2,mask=profmap,assign=pixels[category==1],den=False)
        M = diagonal_operator(px,diag=R.adjoint_times(NpE.inverse_times(R(1))),
                              bare=False)
        M2 = diagonal_operator(px,diag=R2.adjoint_times(NpE2.inverse_times(R2(1))),
                               bare=False)
        E = diagonal_operator(ds,diag=eta_e*sigma_e**2,bare=True)

        Mlmdiag = 0.
        M2lmdiag = 0.

        D = propagator_operator(px,Cl,Mlmdiag,M,1.e-3,imp=True)
        D2 = propagator_operator(px,Cl,M2lmdiag,M2,1.e-3,imp=True)
        RDRdagger = RDRdagger_operator(ds,para=[R,D])
        RD2Rdagger = RDRdagger_operator(ds,para=[R,D2])
        aux = aux_operator(ds,para=[R,D2,R2,NpE2])

        #iterate the estimate of the maps, the Galactic power spectrum, and the
        #correction factors for the error bars and extragalactic variance:
        for i in range(toplevel*10,(toplevel+1)*10):
            
            #calculate foreground map:
            j = R.adjoint_times(NpE.inverse_times(d))
            j2 = R2.adjoint_times(NpE2.inverse_times(d[category==1]))
            m_g = D(j)
            print 'm_g done.'
            np.save(rundir + 'm_g_%02u.npy'%(i+1),m_g)
            
            #calculate foreground map based only on VIP data:
            m_g2 = D2(j2)
            print 'm_g2 done.'
            np.save(rundir + 'm_g2_%02u.npy'%(i+1),m_g2)

            #calculate estimate for the extragalactic contributions:
            m_e = E(NpE.inverse_times(d - R(m_g)))
            print 'm_e done.'
            np.save(rundir + 'm_e_%02u.npy'%(i+1),m_e)
            
            #calculate eta factors:
            D.acc = 0.05
            D2.acc = 0.05
            RDRdagger.para[1] = D
            RD2Rdagger.para[1] = D2
            eta_i, eta_e = calc_eta(d,R,R2,m_g,m_g2,RDRdagger,RD2Rdagger,D2,eta_i,
                                    eta_e,sigma,sigma_e,category,aux,do_eta_e=True)
            np.savetxt(rundir + 'eta_i_%02u'%(i+1),eta_i)
            np.savetxt(rundir + 'eta_e_%02u'%(i+1),np.array([eta_e]))
            E.set_diag(eta_e*sigma_e**2,bare=True)
            newdiag = sigma**2
            newdiag[category==0] += sigma_e**2
            newdiag[category==0] *= eta_i
            newdiag[category==1] += eta_e*sigma_e**2
            NpE.set_diag(newdiag,bare=True)
            NpE2.set_diag(eta_e*sigma_e**2 + sigma[category==1]**2,bare=True)
            M.set_diag(R.adjoint_times(NpE.inverse_times(R(1))),bare=False)
            M2.set_diag(R2.adjoint_times(NpE2.inverse_times(R2(1))),bare=False)
            D.M = M
            D2.M = M2
            print 'etas done.'
            
            #calculate foreground power spectrum:
            Cl = calc_power(D,m_g)
            np.savetxt(rundir + 'Cl_%02u'%(i+1),Cl)
            D.Cl = Cl
            RDRdagger.para[1] = D
            RD2Rdagger.para[1] = D2
            aux.para[1] = D2
            aux.para[3] = NpE2
            print 'Cl done.'
            
            D.acc = 1.e-3
            D2.acc = 1.e-3
            print i + 1

        #estimate the uncertainty for the Galactic reconstruction
        D.acc = 0.05
        Dhat = np.zeros(px.dim())
        for i in range(5):
            Ddiagtemp = D.diag(bare=True,domain=px,nrun=100,nper=1,ncpu=CPU,loop=False)
            np.save(rundir + 'Dhattemp_%03u.npy'%i,Ddiagtemp)
            Dhat += Ddiagtemp
        Dhat /= 5.
        np.save(rundir + 'Dhat_%2i.npy'%((toplevel+1)*10),Dhat)
    
    return 0


if __name__ == '__main__':
    reconstruct()

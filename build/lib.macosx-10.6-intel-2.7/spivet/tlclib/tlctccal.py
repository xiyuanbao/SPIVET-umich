"""
Filename:  tlctccal.py
Copyright (C) 2007-2010 William Newsome
 
This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details, published at 
http://www.gnu.org/copyleft/gpl.html

/////////////////////////////////////////////////////////////////
Description:
  Module containing utility routines for TLC calibration.

    In order for thermochromic liquid crystals to be used
    effectively as an in-situ thermometer for fluids, calibration
    of the liquid crystal hue verus a known temperature must
    be performed.  The hue of TLC's at a given temperature is a
    strong function of viewing angle, and this view angle versus
    hue dependency must be accounted for in the calibration.
    To protect for the possibility that hue is also a weak
    function of depth (eg, when working with fluids that have
    a characteristic color), TlcLIB calibrations can be
    constructed with depth dependence.  Thermochromic calibration
    will then proceed to generate a polynomial function that
    provides temperature as a function of hue, view angle (theta),
    and depth (or z-value).  That is

        T = T(z,theta,hue)
          = c[i,j,k] * z**(i) * theta**(j) * hue**(k)

    where a sum over i,j,k is implied.  The maximum values of i,j,k
    will be set to the order of the user specified polynomial.  See
    calibrate(), below, for more details.

Contents:
  calibrate()
  dn_diT()
  loadtcpa()
  loadtlccal()
  savetlccal()
  bldrpmat()
"""

from spivet import pivlib
from spivet.pivlib import pivutil, pivpgcal, pivpickle, pivlinalg
import tlctc, tlcutil, tlclibc

from numpy import *
from scipy import linalg, optimize


#################################################################
#
def calibrate(tcpa,porder,lmda):
    """
    ----

    tcpa            # lx4 array of calibration points. 
    porder          # Three element tuple specifying the polynomial order.
    lmda            # Smoothing parameter.
    
    ----

    Primary driver routine for thermochromic calibration.  Users
    simply need to provide a calibration point array, the order 
    of the polynomial to use for fitting, and a smoothing parameter
    value, lmda.

    tcpa is an lx4 array of calibration points, where l is the number
    of points.  theta is the angle between two rays: 1) the ray from
    camera to TLC, and 2) the ray pointing in the direction of light
    sheet propagation (ie, the direction unscattered photons would 
    move).  See viewangle() documentation for more details on theta.
        tcpa[:,0] --- Measured temperature value for the data point.
        tcpa[:,1] --- Z-location for the data point.
        tcpa[:,2] --- theta (view angle) for the data point.
        tcpa[:,3] --- Extracted TLC hue for the data point.

    porder should be specified as
        porder[0] --- Order of polynomial for z dependence.
        porder[1] --- Order of polynomial for theta dependence.
        porder[2] --- Order of polynomial for hue dependence.

    Because calibrate() uses a simple polynomial fit instead of more
    complicated mechanisms (eg, Multivariate Adaptive Regression 
    Splines), high order polynomials are subject to subject to
    excessive 'wiggling.'  A simple but effective mechanism to control
    the amount of wiggling is via the use of a roughness penalty for
    the polynomial.  Essentially, the polynomial is fit to the data
    using a modified cost function which consists of two terms: 1) the
    traditional least squares error, and 2) a penalty that increases
    as the curvature of the polynomial increases.  The modified criterion
    becomes
    
        F = || tcpa[:,0] -T(z,theta,hue) ||^2 +lmda*c'Rc
                
    where
    
        R = integral( (del^2 phi)*(del^2 phi') dz dtheta dhue )
    
    is the roughness penalty matrix, and c is the vector of polynomial
    coefficients ( T(z,theta,hue) = c' phi ).  Note that del^2 represents
    the Laplacian, and phi represents the vector of polynomial basis 
    functions (there is one basis function for each fitted coefficient).  
    
    A vast literature exists on Functional Data Analysis, regression, 
    smoothness/roughness parameters, etc.  A good reference is 
    Ramsay:2005.  The implementation here is straightforward and does 
    not attempt to optimize lmda in any way.  As lmda -> infinity, the 
    calibration 'surface' will approach a plane.  Note: Small values of 
    lmda have a big influence.  For our work, a lmda of 0.002 and 
    pcoeff = [1,1,7] produces a smooth, monotonic calibration function 
    with small RMS error.

    The 'SMOOTHED' statistics output during a call to calibrate()
    represent the roughness penalty fit.  The 'LINEAR' results
    represent a fit using un-penalized least squares (ie, lmda = 0).

    Thermochromic calibration parameters are stored in a dictionary
    with the following entries:
        tlccal['pcoeff'] ----- /Float/ Polynomial coefficient array.
        tlccal['porder'] ----- /Integer/ 3 element tuple specifying
                               the calibration polynomial order in
                               z, theta, and hue.
        tlccal['vrange'] ----- /Float/ 3x2 array specifying the range
                               of independent variables for which the
                               calibration is valid.  Computing
                               temperatures using independent variables
                               outside this range will likely yield 
                               grossly inaccurate results!

    Returns [tlccal,err] for the smoothed calibration (ie, using lmda).
        tlccal ----- Thermochromic calibration dictionary.
        err -------- Error between temperatures computed using the
                     tlccal and the measured values.
    """
    print "STARTING: calibration"

    # Initialization.
    porder = array(porder)

    nz  = porder[0] +1
    nth = porder[1] +1
    nh  = porder[2] +1

    zpwr  = array( range( nz ) )
    thpwr = array( range( nth ) )
    hpwr  = array( range( nh ) )  

    npts = tcpa.shape[0]

    amat = zeros((npts,nz*nth*nh),dtype=float)

    # Determine the valid range for later use of calibration.  The
    # polynomial fit should only be used for parameters within these 
    # bounds.
    vrange      = zeros((4,2),dtype=float)
    vrange[0,0] = tcpa[:,1].min()     # z
    vrange[0,1] = tcpa[:,1].max()
    vrange[1,0] = tcpa[:,2].min()     # theta
    vrange[1,1] = tcpa[:,2].max()
    vrange[2,0] = tcpa[:,3].min()     # hue
    vrange[2,1] = tcpa[:,3].max()
    vrange[3,0] = tcpa[:,0].min()     # temperature
    vrange[3,1] = tcpa[:,0].max()

    # Build the amat entries.  The polynomial is ordered such that powers
    # of z vary slowest, followed by theta, and then by hue.  For a P112
    # system, the coefficients would then be:
    #    T(z,th,h) =     a0 +a1*h +a2*h^2 +th*(a3 +a4*h +a5*h^2)
    #               +z*( a6 +a7*h +a8*h^2 +th*(a9 +a10*h +a11*h^2) )
    print " | Building system matrix."
    for n in range(npts):
        pz  = tcpa[n,1]**zpwr
        pth = tcpa[n,2]**thpwr
        ph  = tcpa[n,3]**hpwr

        for si in range(nz):
            sio = si*nth*nh
            for sj in range(nth):
                sjo = sj*nh
                for sk in range(nh):
                    amat[n,sio+sjo+sk] = pz[si]*pth[sj]*ph[sk]    

    # Do the calibration.
    print " | Performing least squares calibration."
    [pcoeff,res,rnk,sng] = linalg.lstsq(amat,tcpa[:,0])

    tlccal = { 'porder':porder, 'pcoeff':pcoeff, 'vrange':vrange }

    # Build the smoothing matrix and form the new system of equations.  
    print " | Smoothing calibration."
    rmat    = bldrpmat(porder,vrange[0:3,:]) 
    sapamat = dot(amat.transpose(),amat) +lmda*rmat
    bvec    = dot(amat.transpose(),tcpa[:,0])
    spcoeff = pivlinalg.dsolve(sapamat,bvec)

    stlccal = { 'porder':porder, 'pcoeff':spcoeff, 'vrange':vrange }

    # Compute the error.
    [tmpa,tcinac] = tlctc.hue2tmp(tcpa[:,1::],tlccal)
    err           = tmpa -tcpa[:,0]
    rmserr        = sqrt( (err*err).mean() )

    [tmpa,tcinac] = tlctc.hue2tmp(tcpa[:,1::],stlccal)
    serr          = tmpa -tcpa[:,0]
    srmserr       = sqrt( (serr*serr).mean() )    
    
    # Print some diagnostic information.
    print " | ----- THERMOCHROMIC CALIBRATION ERROR [degC] -----"
    print " |        LINEAR         SMOOTHED"
    print " | RMS   %13e %13e" % ( rmserr, srmserr )
    print " | MAX   %13e %13e" % ( err.max(), serr.max() )
    print " | MIN   %13e %13e" % ( err.min(), serr.min() )
    print " | STDEV %13e %13e" % ( err.std(), serr.std() )
    print " |"

    print " | EXITING: calibration"
    return [stlccal,serr]   


#################################################################
#
def loadtcpa(ifpath):
    """
    ----

    ifpath          # Path to pivpickled tcpa data.

    ----

    Loads a pivpickled version of the tcpa data.
    """
    tcpa = pivpickle.pklload(ifpath)

    return tcpa


#################################################################
#
def loadtlccal(ifpath):
    """
    ----

    ifpath          # Path to pivpickled calibration dictionary.

    ----

    Loads a pivpickled version of the tlccal dictionary.
    """
    tlccal = pivpickle.pklload(ifpath)
    
    return tlccal


#################################################################
#
def savetlccal(tlccal,ifpath):
    """
    ----

    tlccal          # TLC calibration dictionary to pickle.
    ifpath          # Path to pivpickled calibration dictionary.

    ----

    Saves a pivpickled version of the tlccal dictionary.
    """
    pivpickle.pkldump(tlccal,ifpath)


#################################################################
#
def dn_diT(idvar,dord,dvndx,tlccal):
    """
    ----
    
    idvar          # Independent variable array.
    dord           # Order of derivative.
    dvndx          # Variable index with which to compute derivative.
    tlccal         # Thermochromic calibration dictionary.
    
    ----
    
    Utility function for computing and evaluating the partial derivative 
    of the calibration polynomial:
        (d/dvndx)^dord T(z,theta,hue)
    
    idvar should be an lxm array of points, where l is the number of
    points at which to evaluate the polynomial derivative, and m is
    the number of independent variables in the polynomial function (3 
    for the thermochromic calibration polynomial: z, theta, hue).  The
    variables must be ordered the same as the tlccal parameter
    porder (ie, as z, theta, and hue for the thermochromic polynomial).

    Returns rva, an l-element array containing the results of evaluating
    the derivative at each point in idvar.
    """
    # Initialization.
    pcoeff = tlccal['pcoeff']
    porder = tlccal['porder']

    mpcoeff = pcoeff.copy()
    mpcoeff = mpcoeff.reshape(array(porder) +1)

    mporder        = porder.copy()
    mporder[dvndx] = mporder[dvndx] -dord
    
    if ( not (mporder > -1).all() ):
        return zeros(idvar.shape[0],dtype=float)
    
    ndxmat = indices(array(porder) +1,dtype=float)
    sf     = ndxmat[dvndx,...]
    
    # Modify the coefficients.  Refer to the example polynomial given
    # in calibrate() code with regard to constructing amat for a mental
    # model with which to understand how the following bits work.  That
    # polynomial shows how the coefficients are ordered and hence can be
    # modified when computing a derivative.
    for d in range(dord):
        mpcoeff = mpcoeff*sf       
        sf      = sf -1

    sobj = []
    for v in range( len(porder) ):
        if ( v == dvndx ):
            sobj.append(slice(dord,porder[v]+1))
        else:
            sobj.append(slice(0,porder[v]+1))
    
    mpcoeff = mpcoeff[tuple(sobj)]
    mpcoeff = mpcoeff.reshape(mpcoeff.size)

    # Evaluate the derivative.
    rva = tlclibc.evalmpoly(idvar,mpcoeff,mporder)

    return rva
    

#################################################################
#
def bldrpmat(porder,vrange,nipts=2000):
    """
    ----
    
    porder         # Tuple specifying the polynomial order.
    vrange         # Valid range of polynomial variables.
    nipts=2000     # Number of rougness estimate integration points.
    
    ---- 

    Utility function that computes the roughness penalty matrix:

        rpmat = integral( (del^2 phi)*(del^2 phi') dz dtheta dhue )
    
    where phi is the column vector of polynomial basis functions.
    See Ramsay:2005, Eq. 5.8 for more details.
    
    Returns rpmat, the roughness penalty matrix.
    """
    # Initialization.
    nz  = porder[0] +1
    nth = porder[1] +1
    nh  = porder[2] +1

    nbas = nz*nth*nh
    
    nc = 3  # Hardwire to 3 independent variables.
    
    fvol = vrange[:,1] -vrange[:,0]
    fvol = fvol.prod()
        
    # Build array of function evaluation points.
    random.seed(0)
    pta = empty((nipts,nc),dtype=float)
    for c in range(nc):
        pta[:,c] = (vrange[c,1] -vrange[c,0])*random.rand(nipts) +vrange[c,0]
    random.seed(None)
    
    # Get Laplacian of basis functions.
    dsb = []
    for i in range(nbas):
        coeff    = zeros(nbas,dtype=float)
        coeff[i] = 1.
        
        tlccal = {'porder':porder,'pcoeff':coeff}

        dsbc = dn_diT(pta,2,0,tlccal)
        for v in range(1,nc):
            dsbc = dsbc +dn_diT(pta,2,v,tlccal)
            
        dsb.append(dsbc)
            
    # Build the matrix.
    rpmat = empty((nbas,nbas),dtype=float)
    for i in range(nbas):
        for j in range(nbas):
            rpmat[i,j] = fvol*( dsb[i]*dsb[j] ).mean()
            
    return rpmat
            
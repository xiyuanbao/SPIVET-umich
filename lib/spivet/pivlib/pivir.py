"""
Filename:  pivir.py
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
  Module containing image registration routines for pivlib.

Contents:
  irlk()
  irncc()
  irsctxt()
  irssda()
"""

import pivlibc, pivutil, pivtpsc
from numpy import *
from scipy import linalg, fftpack, ndimage, stats
from spivet import compat

# INAC values.  
irinac = {'ok':0,
          'irlk_maxdisp':10,
          'irlk_linalgerr':11,
          'irlk_mineig':12,
          'irncc_maxdisp':20,
          'irncc_gfit':23,
          'irsctxt_maxdisp':30,
          'irsctxt_linalgerr':31,
          'irsctxt_valueerr':34,
          'irssda_maxdisp':40}


#################################################################
#
def irlk(f1,f2,rbndx,maxdisp,pivdict,rfactor=1.,pinit=None):
    """
    ----
    
    f1             # Frame 1 (greyscale mxn array).
    f2             # Frame 2 (greyscale mxn array).
    rbndx          # Region boundary index (2x2, referenced to frame 1).
    maxdisp        # Max displacement [y,x].
    pivdict        # Config dictionary.
    rfactor=1.     # Relaxation factor.
    pinit=None     # Displacement vector to be used as an initial guess.
    
    ---- 
       
    Performs image registration using the Lucas-Kanade algorithm
    (Lucas:1981) by solving (A'*W*W*A)*v = A'*W*W*ft where 
       A is an lx2 matrix of spatial derivatives,
       W is a diagonal matrix of weights (here set to 1.),
       ft is an vector of length l containing the temporal derivatives,
       and l is the image length (l = m*n).

    irlk() does not compute the full set of affine transforms, but is
    limited to computing translation only.

    NOTE:  irlk() assumes the image has sufficient padding (i.e.,
    don't set bsndx = [0,0]).

    irlk() will not consider any incremental displacements (ie, in
    addition to pinit) larger than maxdisp.  If a computed incremental
    displacement reaches maxdisp, the iterative loop will terminate, 
    p will be set to pinit + maxdisp (with computed direction
    preserved), and inac will be set > 0.

    irlk() provides a means to accelerate convergence by way of rfactor,
    the relaxation factor.  rfactor must be less than 2.0.  If the 
    relaxation factor is too large, the method may diverge. 

    If a pinit vector is passed to irlk(), it will be used as a starting point
    for registration.  Otherwise, irlk() will start with a displacement
    of [0,0].

    Returns [p,inac] where
        p ----- The computed displacements, p=[dy,dx], in moving 
                from f1 to f2.
        inac -- Inaccuracy flag.  Set > 0 when method fails altogether or
                simply doesn't converge.

    NOTE: If the images can't be registered, the method returns 'None' 
    for p.  On the other hand, if the images fail to converge, p will be
    truncated to pinit + maxdisp (with computed direction preserved).
    """

    # Initialization.
    rbndx = array(rbndx)
    rsize = rbndx[:,1] -rbndx[:,0]
    bsndx = rbndx[:,0]
    bendx = rbndx[:,1]

    blen    = rsize[0]*rsize[1] 
    mineig  = pivdict['ir_mineig']
    eps     = pivdict['ir_eps']
    maxits  = pivdict['ir_maxits']
    imthd   = pivdict['ir_imthd']
    iedge   = pivdict['ir_iedge']

    dp = zeros(2,dtype=float)    
    if ( not compat.checkNone(pinit) ):
        ip = array(pinit)
    else:
        ip = zeros(2, dtype=float)

    [apamat, apmat] = pivlibc.irinit(f2.astype(float),bsndx,rsize)

    eig = linalg.eig(apamat,right=False)
    emn = min(eig)
    emx = max(eig)
    eig = [emn, emx]
    if ( emn < mineig ):
        return [None,irinac['irlk_mineig']]  

    # Iterative loop.
    it    = 0
    l2ddp = 10.
    inac  = irinac['ok']
    while ( ( l2ddp > eps ) and ( it < maxits ) ):
        if (    ( abs(dp[0]) >= abs(maxdisp[0]) ) 
             or ( abs(dp[1]) >= abs(maxdisp[1]) ) ):
            inac = irinac['irlk_maxdisp']
            dp   = sign(dp)*array(maxdisp)
            break
            
        if ( (it > 0 ) or (not compat.checkNone( pinit) ) ):
            imbfr = pivutil.imshift(f1,rbndx,ip+dp,imthd,iedge)
        else:
            imbfr = f1[bsndx[0]:bendx[0],bsndx[1]:bendx[1]]

        ft = reshape(imbfr -f2[bsndx[0]:bendx[0],bsndx[1]:bendx[1]],blen)

        bvec = dot(apmat, ft)
        try:
            ddp   = linalg.solve(apamat, bvec)
        except linalg.LinAlgError:
            return [None,irinac['irlk_linalgerr']] 
        l2ddp = linalg.norm(ddp,2)
        dp    = dp +rfactor*ddp

        it = it +1
    return [ip +dp, inac]


#################################################################
#
def irncc(f1,f2,rbndx,maxdisp,pivdict,pinit=None):
    """
    ----
    
    f1             # Frame 1 (greyscale mxn array).
    f2             # Frame 2 (greyscale mxn array).
    rbndx          # Region boundary index (2x2, referenced to frame 1).
    maxdisp        # Max displacement [y,x].
    pivdict        # Config dictionary.
    pinit=None     # Displacement vector to be used as an initial guess.
    
    ----
        
    Performs image registration using normalized cross-correlation. 
    The normalized cross-correlation coefficient is insensitive to
    variations in illumination between frames and is not impacted by
    local intensity variations of the search frame (f2) (ref. Brown:1992).
    Although slower than cross correlation, images consisting of broad, 
    undulating intensity that more resembles a fabric than a collection of 
    bright dots can easily be registered using irncc().  

    f1 will be used as the template and will be shifted by pinit, if
    supplied, prior to computing the correlation.

    Returns [p,inac,cmax] where
        p ----- The computed displacements, p=[dy,dx], in moving 
                from f1 to f2.
        inac -- Inaccuracy flag.  Set > 0 when gaussian fit can't be 
                performed or fails.
        cmax -- Maximum value of the NCC measure.  A value of 1.0
                indicates a perfect match, with cmax decreasing as the
                match quality decreases.

    NOTE: If the gaussian fit fails, the method returns 'None' 
    for p.  Otherwise if a gaussian fit can't be performed (because
    the computed displacement equals maxdisp in one or both
    directions), the best guess to p is returned and inac is set.
    """

    # Initialization
    rbndx = array(rbndx)
    rsize = rbndx[:,1] -rbndx[:,0]
    
    maxits  = pivdict['ir_maxits']
    eps     = pivdict['ir_eps']
    imthd   = pivdict['ir_imthd']
    iedge   = pivdict['ir_iedge']

    maxdisp = array(maxdisp)

    p     = zeros(2, dtype=float)
    mxndx = zeros(2, dtype=int)

    prsize    = [0,0]
    prsize[0] = 1 +2*maxdisp[0]
    prsize[1] = 1 +2*maxdisp[1]

    inac = irinac['ok']

    # Get copy of the template block.
    if ( compat.checkNone(pinit) ):
        f1bfr = f1[rbndx[0,0]:rbndx[0,1],
                   rbndx[1,0]:rbndx[1,1]].copy()
        ip = zeros(2,dtype=float)
    else:
        f1bfr = pivutil.imshift(f1,rbndx,pinit,imthd,iedge)
        ip    = array(pinit)

    # Compute the correlation coefficient matrix.
    coeff = pivlibc.irncc_core(f1bfr,
                               f2,
                               rbndx.reshape(4),
                               maxdisp.astype(float))

    # Find the maximum.
    mx       = argmax(coeff)
    mxndx[0] = int(mx/prsize[1]) 
    mxndx[1] = int(mod(mx,prsize[1]))

    # Fit to gaussian.
    if ( ( mxndx[0] == 0 ) or ( mxndx[0] == prsize[0] -1 ) ):
        p[0] = ip[0] +mxndx[0] -maxdisp[0]
        inac = irinac['irncc_maxdisp']
    else:
        svec = pivutil.gfit(
            coeff[(mxndx[0]-1):(mxndx[0]+2),mxndx[1]],
            maxits,
            eps)
        if ( compat.checkNone(svec) ):
            return [None,irinac['irncc_gfit'],0.]
        p[0] = ip[0] +mxndx[0] +svec[1] -maxdisp[0]
    if ( ( mxndx[1] == 0 ) or ( mxndx[1] == prsize[1] -1 ) ):
        p[1] = ip[1] +mxndx[1] -maxdisp[1]
        inac = irinac['irncc_maxdisp']
    else:
        svec = pivutil.gfit(
            coeff[mxndx[0],(mxndx[1]-1):(mxndx[1]+2)],
            maxits,
            eps)
        if ( compat.checkNone(svec) ):
            return [None,irinac['irncc_gfit'],0.]
        p[1] = ip[1] +mxndx[1] +svec[1] -maxdisp[1]

    return [ p, inac, coeff[mxndx[0],mxndx[1]] ]


#################################################################
#
def irsctxt(f1,f2,rbndx,maxdisp,pivdict):
    """
    ----
    
    f1             # Frame 1 (greyscale mxn array).
    f2             # Frame 2 (greyscale mxn array).
    rbndx          # Region boundary index (2x2, referenced to frame 1).
    maxdisp        # Max displacement [y,x].
    pivdict        # Config dictionary.
    
    ----
        
    Performs image registration using thin-plate splines (TPS), shape
    contexts, and a spring model.  See documentation for esttpswarp() 
    for details on the methods used.

    The TPS model allows a smoothly varying displacement field to be
    constructed for the region of interest.  This non-rigid displacement
    field differs from the displacement-only fields of irssda() and irncc()
    that are taken to be uniform across the entire region of interest.
    Regardless, irsctxt() is expected to return a single p (displacement)
    vector for the image block under consideration.  This aggregate
    displacement vector is computed as follows.  After the TPS warp for 
    the region of interest is estimated, the coordinates of all f1 pixels
    within the region of interest are warped using the TPS warp.  The
    p value is then set to the average displacement.

    Returns [p,inac] where
        p ----- The computed displacements, p=[dy,dx], in moving 
                from f1 to f2.
        inac -- Inaccuracy flag.  inac will be set greater than 0 only
                if an exception is thrown during the call to esttpswarp(),
                or if the final p value exceeds maxdisp.
                
    """
    # Initialization
    imthd   = pivdict['ir_imthd']
    iedge   = pivdict['ir_iedge']
    csrp    = pivdict['ir_tps_csrp']
    ithp    = pivdict['ir_tps_ithp']
    wsize   = pivdict['ir_tps_wsize']
    sdmyf   = pivdict['ir_tps_sdmyf']
    alpha   = pivdict['ir_tps_alpha']
    beta    = pivdict['ir_tps_beta']
    csize   = pivdict['ir_tps_csize']
    nits    = pivdict['ir_tps_nits']
    scit    = pivdict['ir_tps_scit']
    annl    = pivdict['ir_tps_annl']

    maxdisp = array(maxdisp)
    
    rbndx = array(rbndx)
    rsize = rbndx[:,1] -rbndx[:,0]

    xrbndx      = rbndx.copy()
    xrbndx[:,0] = xrbndx[:,0] -maxdisp
    xrbndx[:,1] = xrbndx[:,1] +maxdisp
    xrsize      = xrbndx[:,1] -xrbndx[:,0]
    
    srgn = f2[xrbndx[0,0]:xrbndx[0,1],
              xrbndx[1,0]:xrbndx[1,1]]
    
    inac = irinac['ok'] 

    # Determine the warp parameters.
    try:
        tpw  = pivutil.esttpswarp(f1,f2,xrbndx,
                                  csrp=csrp,
                                  ithp=ithp,
                                  wsize=wsize,
                                  sdmyf=sdmyf,
                                  alpha=alpha,
                                  beta=beta,
                                  csize=csize,
                                  nits=nits,
                                  scit=scit,
                                  annl=annl)
    except linalg.LinAlgError:
        return [None,irinac['irsctxt_linalgerr']]
    except ValueError:
        return [None,irinac['irsctxt_valueerr']]
    
    # Compute p estimate.
    ndxmat = indices(rsize,dtype=float)
    npts   = rsize[0]*rsize[1]
    
    yvec = ndxmat[0,...].reshape(npts) +maxdisp[0]
    xvec = ndxmat[1,...].reshape(npts) +maxdisp[1] 
    tpts = array( [yvec,xvec] ).transpose()
    
    wpts = tpw.xfrm(tpts)
    dlta = wpts -tpts
    p    = dlta.mean(0)
    if ( ( p > maxdisp ).any() ):
        inac = irinac['irsctxt_maxdisp']
    
    return [p,inac]


#################################################################
#
def irssda(f1,f2,rbndx,maxdisp,pivdict,pinit=None):
    """
    ----
    
    f1             # Frame 1 (greyscale mxn array).
    f2             # Frame 2 (greyscale mxn array).
    rbndx          # Region boundary index (2x2, referenced to frame 1).
    maxdisp        # Max displacement [y,x].
    pivdict        # Config dictionary.
    pinit=None     # Displacement vector to be used as an initial guess.
    
    ----
        
    Performs image registration using an SSDA-type similarity measure,
    ref Brown:1992.

    f1 will be used as the template and will be shifted by pinit, if
    supplied, prior to computing the correlation.

    Returns [p,inac,cmin] where
        p ----- The computed displacements, p=[dy,dx], in moving 
                from f1 to f2.
        inac -- Inaccuracy flag.  
        cmin -- Minimum value of SSDA measure.  A value of 0.0 indicates
                a perfect match, with cmin increasing as the match quality
                decreases.

    NOTE: irssda() only registers images to a precision of 1 pixel.

    NOTE: After registering with irssda(), irncc() should be called
    to improve accuracy.
    """
    # Initialization
    rbndx = array(rbndx)
    rsize = rbndx[:,1] -rbndx[:,0]
    
    imthd   = pivdict['ir_imthd']
    iedge   = pivdict['ir_iedge']

    maxdisp = array(maxdisp)

    p     = zeros(2, dtype=float)
    mxndx = zeros(2, dtype=int)

    prsize    = [0,0]
    prsize[0] = 1 +2*maxdisp[0]
    prsize[1] = 1 +2*maxdisp[1]

    inac = irinac['ok']

    # Get copy of the template block.
    if ( compat.checkNone(pinit) ):
        f1bfr = f1[rbndx[0,0]:rbndx[0,1],
                   rbndx[1,0]:rbndx[1,1]].copy()
        ip = zeros(2,dtype=float)
    else:
        f1bfr = pivutil.imshift(f1,rbndx,pinit,imthd,iedge)
        ip    = array(pinit)

    coeff = pivlibc.irssda_core(f1bfr,
                                f2,
                                rbndx.reshape(4),
                                maxdisp.astype(float))

    # Find the minimum.
    mx       = argmin(coeff)
    mxndx[0] = int(mx/prsize[1]) 
    mxndx[1] = int(mod(mx,prsize[1]))
        
    # Convert to translation.
    if ( ( mxndx[0] == 0 ) or ( mxndx[0] == prsize[0] -1 ) ):
        inac = irinac['irssda_maxdisp']

    p[0] = ip[0] +mxndx[0] -maxdisp[0]

    if ( ( mxndx[1] == 0 ) or ( mxndx[1] == prsize[1] -1 ) ):
        inac = irinac['irssda_maxdisp']

    p[1] = ip[1] +mxndx[1] -maxdisp[1]

    return [ p, inac, coeff[mxndx[0],mxndx[1]] ]

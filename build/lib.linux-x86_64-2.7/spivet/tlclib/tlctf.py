"""
Filename:  tlctf.py
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
  Module containing temperature field processing routines.

Contents:
  tfcomp()

"""

from numpy import *

from spivet import pivlib, flolib
from spivet.pivlib import pivutil, pivpgcal
import tlctc, tlcutil
from spivet import compat

#################################################################
#
def tfcomp(chnlh, swz, pivdict, pmsk=None):
    """
    ----

    chnlh           # Dewarped hue channel.
    swz             # The stationary z-coordinate of the image plane.
    pivdict         # Config dictionary.
    pmsk=None       # Particle mask array of same shape as chnlh.

    ----

    tfcomp() is the primary driver routine for extracting temperatures
    from TLC hue values.  

    Hues are converted to temperatures in a two step process.  
       1) The hue image is subdivided into blocks, and the average 
          hue for each block is computed using only valid particles
          from the particle mask, pmsk, if provided.  If no particle
          mask is provided, then the average hue will be computed using
          all pixels in the block.
       2) The average hue values from (1) are converted to temperature 
          values by calling hue2tmp(). 

    TLC temperature versus hue calibrations can be dependent upon
    how much fluid lies between the illuminated sheet of crystals and
    the cameras.  For a 3D system similar to that at UofM where the
    the cameras and lightsheet are motion controlled, two world
    coordinate systems exist.  One moves with the lightsheet, the other
    is stationary and fixed to 'laboratory coordinates.'  swz refers
    to the fixed world coordinates.  If the TLC calibration has no
    depth dependence, then swz can be set to zero (or any other value).  
    Otherwise, it must take on a value using the same, fixed world origin
    used during TLC calibration.

    A mask of valid particles can be passed to limit the pixels in the
    hue channel that will be used to compute the temperature for a 
    given block.  If the value of pmsk is greater than zero for a 
    pixel, then that chnlh pixel will be retained during temperature 
    computation.  A convenience function, tlcmask(), is provided to 
    compute such particle masks.  Its use is highly encouraged to limit
    errors caused by RGB pixels having low saturation or intensity.

    At present, tfcomp() can only handle thermochromic calibration
    with one camera.

    Given the nature of TLC's and the need to compute an average
    color over a reasonably large block size, tfcomp() does not
    sub-divide blocks using gp_bsdiv during temperature computation
    unless the pivdict parameter tc_interp = False.  When 
    tc_interp = True, tfcomp() will compute the temperature using blocks
    of size gp_bsize and then interpolate the coarse block result into 
    any sub-blocks.  

    NOTE: tfcomp() does make full use of gp_bolap.

    Returns [tf,tfINAC], two PIVVar objects.  s,t below are the
    number of blocks in the y and x directions respectively.
        tf ------ 1 x 1 x s x t PIVVar object containing the computed
                  temperature.  Variable name will be set to 'T'
        tfINAC -- 1 x 1 x s x t PIVVar object containing the inaccuracy
                  flag for tfcomp() results.  The possible tfINAC values 
                  are essentially those of hue2tmp() with the additional
                  incremental value defined:
                      64 --- No valid particles found in block.
    """

    print "STARTING: tfcomp"

    # Initialization.
    rbndx     = pivdict['gp_rbndx']
    bsize     = pivdict['gp_bsize']
    bolap     = pivdict['gp_bolap']
    bsdiv     = pivdict['gp_bsdiv']
    mmpx      = pivdict['pg_wicsp'][0]['mmpx']
    wos       = pivdict['pg_wicsp'][0]['wos']
    ilvec     = pivdict['tc_ilvec']
    interp    = pivdict['tc_interp']
    tlccal    = pivdict['tc_tlccal']
    tlccam    = pivdict['tc_tlccam']
    tlccamcal = pivdict['pg_camcal'][tlccam]

    imdim = chnlh.shape
    rbndx = array(rbndx)

    [rsize,lbso,hbso,lnblks,hnblks] = pivutil.getblkprm(rbndx,bsize,
                                                        bolap,bsdiv)

    htnblks = hnblks[0]*hnblks[1]

    pbsndx = zeros(2,dtype=int)
    pbendx = pbsndx.copy()
    
    if ( interp and bsdiv > 1 ):  
        print ' | nyblks '+str(lnblks[0])
        print ' | nxblks '+str(lnblks[1])
        
        ptnblks = lnblks[0]*lnblks[1]
        pbsize  = bsize
        pnblks  = lnblks  
        pbso    = lbso
    else:
        print ' | nyblks '+str(hnblks[0])
        print ' | nxblks '+str(hnblks[1])  

        ptnblks = hnblks[0]*hnblks[1]
        pbsize  = array(bsize)/bsdiv
        pnblks  = hnblks
        pbso    = hbso

    iva   = zeros((ptnblks,3),dtype=float)

    tf     = pivlib.PIVVar((1,1,hnblks[0],hnblks[1]),"T","DEGC")
    tfINAC = pivlib.PIVVar((1,1,hnblks[0],hnblks[1]),"TINAC","NA",dtype=int)

    if ( compat.checkNone(pmsk) ):
        pmsk = ones(imdim,dtype=float)
    
    # Compute coordinates of block centers in mobile world coordinates.
    ndxmat = indices((pnblks[0],pnblks[1]))
    yv = ndxmat[0,:,:]
    xv = ndxmat[1,:,:]

    yv = (yv*(pbsize[0] -bolap[0]) +rbndx[0,0] +pbsize[0]/2.)*mmpx +wos[0]
    xv = (xv*(pbsize[1] -bolap[1]) +rbndx[1,0] +pbsize[1]/2.)*mmpx +wos[1]

    zv = zeros(ptnblks,dtype=float)
    yv = yv.reshape(ptnblks)
    xv = xv.reshape(ptnblks)
 
    wca = array([zv,yv,xv]).transpose()

    # Get view angles and build independent variable array.
    iva[:,0] = swz    
    iva[:,1] = tlctc.viewangle(wca,tlccamcal,ilvec)
    
    # Grab the average hues.
    print " | Computing hue."
    iva        = iva.reshape((pnblks[0],pnblks[1],3))
    [bh,hinac] = tlcutil.xtblkhue(chnlh,pmsk,rbndx,pnblks,pbsize,pbso)
    iva[:,:,2] = bh

    # Determine temperatures.
    print " | Calculating temperatures."
    iva       = iva.reshape((ptnblks,3))
    [tfa,tfi] = tlctc.hue2tmp(iva,tlccal)

    hinac = hinac.reshape(ptnblks)
    tfi   = tfi.reshape(ptnblks)
    tfi   = where(hinac,tfi+64,tfi)
    
    tfa = tfa.reshape((1,1,pnblks[0],pnblks[1]))
    tfi = tfi.reshape((1,1,pnblks[0],pnblks[1]))
    
    # Store results in PIVVar's.
    if ( interp and bsdiv > 1 ):        
        oset          = (bsdiv -1.)/(2.*bsdiv)
        icrd          = indices([1,hnblks[0],hnblks[1]],dtype=float)
        icrd[1:3,...] = icrd[1:3,...]/bsdiv -oset
        
        icrd = icrd.reshape([3,htnblks])
        icrd = icrd.transpose()
        
        tfa = flolib.svinterp(tfa,icrd)
        tfa = tfa.reshape([1,1,hnblks[0],hnblks[1]])
        
        tfi = tfi.repeat(bsdiv,axis=2)
        tfi = tfi.repeat(bsdiv,axis=3)
    
    tf[:,:,:,:]     = tfa[:,:,:,:]
    tfINAC[:,:,:,:] = tfi[:,:,:,:]

    print " | EXITING: tfcomp"
    return [tf,tfINAC]


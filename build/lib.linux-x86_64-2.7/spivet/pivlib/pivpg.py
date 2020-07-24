"""
Filename:  pivpg.py
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
  pivpg contains general photogrammetric functions.

Contents:
  dscwrld()
  loadwicsp()
  prjim2wrld()
  wrld2imcsp()
"""

from numpy import *

import pivpgcal
import pivutil
import pivpickle
from spivet import compat

#################################################################
#
def dscwrld(rbndx,camcal):    
    """
    ----
    
    rbndx          # Analyzed region boundary index for image (2x2).
    camcal         # Camera calibration dictionary.
    
    ----
        
    Discretizes world space into pixels.

    dscwrld() automatically discretizes world space to ensure
    that no image data is lost (ie, that world pixels are small
    enough to correspond to, at most, one image pixel).  The
    discretization of world space is carried out by projecting
    the coordinates of the image corner pixels onto world space.
    With the world corner coordinates determined, four mm/px
    candidates are computed using forward differences in y and x.
    The smallest value is then selected and multiplied by a scale
    factor (to help further ensure no data is lost).

    Returns a dictionary, wdsc:
        wdsc['wpta'] ---- /Float/ lx3 array of world (y,x) spatial
                          coordinates. [mm]
        wdsc['wpxa'] ---- /Int/ lx2 array of world (y,x) pixel 
                          coordinates. [pixels]
        wdsc['wpxdim'] -- /Int/ 2-element array specifying the number
                          of pixels spanning world space in (y,x).
        wdsc['mmpx'] ---- /Float/ Discretization size of world space.
                          [mm/pixel]
    Note: l = wpxdim[0]*wpxdim[1].
    """
    # Initialization.
    rbndx = array(rbndx)
    rsize = rbndx[:,1] -rbndx[:,0]

    wdsc = {}

    # Tuning parameters.
    mmpxsf = 0.8     # MM/PX scale factor. 

    # Determine suitable mm/px.  In the image plane, the coordinates for
    # rbndx form a rectangle.  When the image coordinates for rbndx are
    # projected into world coordinates, they form the corners of a general
    # quadrilateral.  Candidate mm/px values are computed by normalizing 
    # the length for each side of this quadrilateral in world coordinates
    # with the corresponding number of pixels in the image plane (ie, the
    # length of the corresponding rbndx rectangle side in image coordinates).
    # To prevent any loss of information during projection, the smallest
    # of the four candidate mm/px values is chosen and further scaled by
    # mmpxsf.
    impts = array([[rbndx[0,0],rbndx[1,0]],[rbndx[0,0],rbndx[1,1]],
                   [rbndx[0,1],rbndx[1,0]],[rbndx[0,1],rbndx[1,1]]])
    cwpts = pivpgcal.im2wrld(impts,camcal)

    cwy = cwpts[:,1].reshape((2,2))
    cwx = cwpts[:,2].reshape((2,2))

    rrsz0 = sqrt( ( cwy[1,:] -cwy[0,:] )**2 +( cwx[1,:] -cwx[0,:] )**2 )
    rrsz1 = sqrt( ( cwy[:,1] -cwy[:,0] )**2 +( cwx[:,1] -cwx[:,0] )**2 )

    mmpxrsz0 = rrsz0/rsize[0]
    mmpxrsz1 = rrsz1/rsize[1]

    mmpx = mmpxsf*min(mmpxrsz0.min(), mmpxrsz1.min())
    
    # Determine the number of pixels required to span world space.
    nypix = (cwy.max() -cwy.min())/mmpx
    nxpix = (cwx.max() -cwx.min())/mmpx
    
    nypix = int( ceil( nypix ) )
    nxpix = int( ceil( nxpix ) )
    
    npts = nypix*nxpix

    # Get coordinates of world pixels.
    wndxm = indices((nypix,nxpix))
    yndx  = wndxm[0,:,:].reshape(npts)
    xndx  = wndxm[1,:,:].reshape(npts)
    
    wpty = cwy.min() +mmpx*yndx
    wptx = cwx.min() +mmpx*xndx
    
    wpta = zeros((npts,3),dtype=float)
    wpta[:,1] = wpty
    wpta[:,2] = wptx

    # Set up the dictionary.
    wdsc['wpta']   = wpta
    wdsc['wpxa']   = array( [yndx,xndx] ).transpose() 
    wdsc['wpxdim'] = array( [nypix,nxpix] )
    wdsc['mmpx']   = mmpx

    return wdsc


#################################################################
#
def loadwicsp(ifpath):
    """
    ----
    
    ifpath          # Path to wicsp file.
    
    ----
        
    Convenience function to load a pivpickled wicsp.
    """
    wicsp = pivpickle.pklload(ifpath)

    return wicsp
    

#################################################################
#
def prjim2wrld(imin,wicsp):
    """
    ----
    
    imin           # Floating point input image.
    wicsp          # World to image correspondence dictionary.
    
    ----
    
    Projects and image onto the z=0 world plane using the
    world discretization stored in wicsp.  prjim2wrld() effectively
    removes all distortion from an image.

    Returns the projected image.
    """
    # Initialization.
    ica    = wicsp['ica']
    isa    = wicsp['isa']
    wpxa   = wicsp['wpxa']
    wpxdim = wicsp['wpxdim']
    
    wim = zeros(wpxdim,dtype=float)

    # Interpolate the input image.
    wdata = pivutil.pxshift(imin,ica,isa)

    # Store the data in the world image.
    wim[wpxa[:,0],wpxa[:,1]] = wdata

    return wim


#################################################################
#
def wrld2imcsp(rbndx,camcal,wdsc,ofpath = None): 
    """
    ----
    
    rbndx           # Analyzed region boundary index for image (2x2).
    camcal          # Camera calibration dictionary.
    wdsc            # World discretization dictionary.
    ofpath = None   # Path to output file for wicsp.
    
    ----
        
    Computes the correspondence (ie, mapping) between world and 
    image coordinates allowing easy, repetitive projection of
    of image data onto the z=0 world plane.

    When projecting an image onto the world plane, the world
    plane must be discretized in a similar fashion to the image
    plane.  As long as the camera calibration doesn't change,
    the relation between a pixel in world space and image
    space is obviously constant.  Technically, the correspondence 
    between the image and world planes is already known via the
    camera calibration, but the camera calibration provides a 
    high level, general function.  As discussed in pivpgcal, the
    conversion of world coordinates to image coordinates is very
    time consuming.  Therefore, significant computational economy 
    can be realized by computing the world to image correspondence 
    on a pixel by pixel basis once and storing the results for future 
    use.

    Returns wicsp, a dictionary containing:
        wicsp['ica'] ----- /Int/ lx2 array of integer parts of image 
                           (v,u) coordinates corresponding to world
                           pixel coordinates.
        wicsp['isa'] ----- /Float/ lx2 array of negatve fractional 
                           parts of image (v,u) coordinates corresponding 
                           to world pixel coordinates.  The fractional
                           parts are negative so that they can be fed
                           directly to pxshift().
        wicsp['wos'] ----- /Float/ 2-element array specifiying spatial
                           coordinates (in mm) of world pixel 0,0.
        wicsp['wpta'] ---- /Float/ lx3 array of world (y,x) spatial
                           coordinates. [mm]
        wicsp['wpxa'] ---- /Int/ lx2 array of world (y,x) pixel 
                           coordinates. [pixels]
        wicsp['wpxdim'] -- /Int/ 2-element array specifying the number
                           of pixels spanning world space in (y,x).
        wicsp['mmpx'] ---- /Float/ Discretization size of world space.
                           [mm/pixel]
    Note: m = wpxdim[0]*wpxdim[1].
    """
    print "STARTING: wrld2imcsp"
    # Initialization.
    rbndx = array(rbndx)

    wpta   = wdsc['wpta']
    wpxdim = wdsc['wpxdim']
    mmpx   = wdsc['mmpx']

    wptz   = wpta[:,0]
    wpty   = wpta[:,1]
    wptx   = wpta[:,2]

    yndx   = wdsc['wpxa'][:,0]
    xndx   = wdsc['wpxa'][:,1]

    wos    = zeros(2,dtype=float)
    wos[0] = wpta[0,1]
    wos[1] = wpta[0,2]

    wicsp = {}

    print " | World mm/pixel: %f" % mmpx
    print " | World pixels: (%i, %i)" % (wpxdim[0],wpxdim[1])

    # Project world pixel coordinates onto image plane.
    print " | Projecting world pixel coordinates onto image plane."
    ipta = pivpgcal.wrld2im(wpta[:,:],camcal)
    vim = ipta[:,0]
    uim = ipta[:,1]
    
    icv = floor(vim)     # Integer part of image pixel coordinate.
    icu = floor(uim)
    
    isv = icv -vim       # Negative fractional part of image pixel coordinate.
    isu = icu -uim
    
    icv = icv.astype(int)
    icu = icu.astype(int)

    # Keep only those values that correspond to valid image coordinates,
    # except image coordinates that form the outer perimeter of the
    # image (1 pixel wide on top and left, 2 pixels wide on bottom and
    # right).  Protects for pxshift() and bicubic interpolation.
    msk  = ( icv < (rbndx[0,1] -1) )*( icv > rbndx[0,0] )
    icv  = compress(msk,icv)
    icu  = compress(msk,icu)
    isv  = compress(msk,isv)
    isu  = compress(msk,isu)
    wptz = compress(msk,wptz)
    wpty = compress(msk,wpty)
    wptx = compress(msk,wptx)
    yndx = compress(msk,yndx)
    xndx = compress(msk,xndx)
    
    msk  = ( icu < (rbndx[1,1] -1) )*( icu > rbndx[1,0] )
    icv  = compress(msk,icv)
    icu  = compress(msk,icu)
    isv  = compress(msk,isv)
    isu  = compress(msk,isu)
    wptz = compress(msk,wptz)
    wpty = compress(msk,wpty)
    wptx = compress(msk,wptx)
    yndx = compress(msk,yndx)
    xndx = compress(msk,xndx)

    # Set up the dictionary.
    wicsp['ica']    = array( [icv,icu] ).transpose()
    wicsp['isa']    = array( [isv,isu] ).transpose()
    wicsp['wos']    = wos
    wicsp['wpta']   = array( [wptz,wpty,wptx] ).transpose()
    wicsp['wpxa']   = array( [yndx,xndx] ).transpose() 
    wicsp['wpxdim'] = wpxdim
    wicsp['mmpx']   = mmpx

    # Store the dictionary.
    if ( not compat.checkNone(ofpath) ):
        pivpickle.pkldump(wicsp,ofpath)

    print " | EXITING: wrld2imcsp"

    return wicsp


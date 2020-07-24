"""
Filename:  tlctc.py
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
  Module containing general thermochromic functions.

Contents:
  hue2tmp()
  viewangle()
"""

from spivet.pivlib import pivpgcal
import tlclibc

from numpy import *


#################################################################
#
def hue2tmp(idvar, tlccal):
    """
    ----

    idvar           # Independent variables to convert to temperature.
    tlccal          # Thermochromic calibration dictionary.

    ----

    Computes a temperature value from the (z,theta,hue) values of
    idvar.  idvar can be an lx3 array of points, where l is the
    number of points to be converted.  Each point must be ordered as
    (z,theta,hue).

    NOTE: z is the z-coordinate in stationary world coordinates.

    Returns [tmpa,tcinac] where
        tmpa ----- l element array of temperature values.
        tcinac --- l element array indicating whether the independent 
                   variables used in the computation of the temperature
                   were outside the valid range.  The value of tcinac
                   is a sum of the following possible states:
                       0 -- Independent variable valid.
                       1 -- z-coordinate outside valid range.
                       2 -- theta outsize valid range.
                       4 -- hue outside valid range.
    """
    # Initialization.
    porder = tlccal['porder']
    pcoeff = tlccal['pcoeff']
    vrange = tlccal['vrange']

    ida = array(idvar)
    if ( ida.size == 3 ):
        ida = ida.reshape((1,3))

    npts = ida.shape[0]

    nz  = porder[0] +1
    nth = porder[1] +1
    nh  = porder[2] +1

    zpwr  = array( range( nz ) )
    thpwr = array( range( nth ) )
    hpwr  = array( range( nh ) )

    tmpa   = zeros(npts,dtype=float)
    tcinac = zeros(npts,dtype=int) 

    # Compute the temperatures.
    tmpa = tlclibc.evalmpoly(idvar,pcoeff,porder)

    for i in range(3):
        if ( porder[i] > 0 ):
            msk = ( ida[:,i] < vrange[i,0] ) + ( ida[:,i] > vrange[i,1] )    
            tcinac[msk] = tcinac[msk] +2**i

    return [tmpa,tcinac]


#################################################################
#
def viewangle(wrldpt, camcal, ilvec):
    """
    ----

    wrldpt          # Point in world space (z,y,x) coordinates.
    camcal          # Camera calibration used for thermochromic calibration.
    ilvec           # Illumination unit vector (z,y,x).

    ----

    Utility function to compute the liquid crystal view angle.

    View angle is taken to be the angle in world-coordinates between 
    the rays: a) camera and the particle, and b) ilvec. 

    ilvec points from the light source along the direction of unscattered
    ray propagation (ie, without hitting any tracer particles and being
    scattered toward the camera).  For most purposes, the illumination
    source can be assumed well-collimated and ilvec taken as a constant.
    For our setup, a lightsheet is used that propagates along the
    negative x-axis, so ilvec = (0,0,-1).  

    wrldpt can be an lx3 array of points where l is the number of 
    points to be converted.  Each point must be ordered as (z,y,x)

    Returns an array of length l containing the view angles for
    wrldpt's.
    """
    # Initialization.
    ilvec = array(ilvec)
    wpa   = array(wrldpt)
    if ( wpa.size == 3 ):
        wpa = wpa.reshape((1,3))

    npts = wpa.shape[0]

    ilvec = ilvec/sqrt( (ilvec*ilvec).sum() )

    # Compute camera center in world coordinates (wcc) and the view
    # angle.  
    wcc  = linalg.solve(camcal['Rmat'],-camcal['T'])    
    wcc  = repeat(wcc,npts)
    wcc  = wcc.reshape((3,npts)).transpose()
    pray = wpa -wcc  # Points from camera to particle.

    nrm = 1./sqrt( (pray*pray).sum(axis=1) )
    nrm = repeat(nrm,3)
    nrm = nrm.reshape(pray.shape)

    pray = nrm*pray

    theta = arccos( dot(pray,ilvec) )

    return theta

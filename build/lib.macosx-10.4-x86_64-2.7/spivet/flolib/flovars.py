"""
Filename:  flovars.py
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
  Module containing routines for computing flow variables from
  PIV data.

Contents:
  vorticity()
"""

from spivet.pivlib import pivdata
import floutil

from numpy import *

#################################################################
#
def vorticity(uvar,cellsz):
    """
    ----
    
    uvar          # PIVVar containing velocity vectors.
    cellsz        # 3 element tuple containing the grid cell size.
    
    ----
        
    Computes the vorticity field from the velocity vectors in
    uvar.

        omega = curl( uvar )

    cellsz contains the grid cell size along the (z,y,x)
    axes.

    Returns a PIVVar, omega, containing the vorticity.  Variable
    name will be set to 'VORT' with units of '1_S'.
    """
    # Initialization.
    if ( not isinstance(uvar,pivdata.PIVVar) ):
        print "ERROR: uvar must be a PIVVar."
        return

    omega = pivdata.PIVVar(uvar.shape,"VORT","1_S")

    # Get the derivatives.
    dwdx = floutil.d_di(uvar[0,:,:,:],2,cellsz[2])
    dwdy = floutil.d_di(uvar[0,:,:,:],1,cellsz[1])
    dvdx = floutil.d_di(uvar[1,:,:,:],2,cellsz[2])
    dudy = floutil.d_di(uvar[2,:,:,:],1,cellsz[1])

    if ( uvar.shape[1] > 1 ):
        dvdz = floutil.d_di(uvar[1,:,:,:],0,cellsz[0])        
        dudz = floutil.d_di(uvar[2,:,:,:],0,cellsz[0])

    # Compute the vorticity.
    omega[0,:,:,:] = dvdx -dudy

    if ( uvar.shape[1] > 1 ):
        omega[1,:,:,:] = dudz -dwdx
        omega[2,:,:,:] = dwdy -dvdz

    return omega

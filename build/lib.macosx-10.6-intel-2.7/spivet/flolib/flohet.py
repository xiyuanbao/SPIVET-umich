"""
Filename:  flohet.py
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
  Module providing facilities for the estimation of heterogeneity
  destruction.  A general flow can distort an initially ball-shaped
  region into a complex shape.  An example of such action would be
  the streaks formed in a viscous fluid as a dyed volume of the fluid
  is stirred into a larger volume.  As the dyed fluid is mechanically
  stirred and the striations become thinner and thinner, the dyed
  material will begin to diffuse more effectively into the undyed 
  fluid.  Hence the mechanical stirring can increase the effectiveness
  of diffusion by reducing the characteristic length scale of the
  material that is diffusing.  
  
  flohet estimates the destructive potential of a flow field by 
  computing the non-dimensional radius of heterogeneity that would be
  destroyed under a time-averaged, diffusion-only analog of the
  real flow.  Imagine releasing a ball of dyed fluid into a flow.
  Provided the initial ball of dyed fluid is small enough, the action 
  of a general flow will transform the infinitesimal ball into an 
  ellipsoidal shape.  Clearly some flows are unsteady, and consequently 
  may stretch the ball in one direction before compressing the resulting 
  ellipsoid back into a ball.  Regardless, for diffusion to destroy a 
  heterogeneity, the flow must maintain the inclusion in a particular 
  state for a sufficiently long period of time.  The flohet module
  considers the time-averaged shape of the infinitesimal ellipsoid 
  and determines the corresponding maximum non-dimensional radius, 
  a*, of the initial inclusion that will fall below a user specified
  concentration under the action of diffusion only.  To first order,
  then, one could estimate that all inclusions with initial radius
  smaller than a* would be destroyed under the action of the flow.     
  
  To use the module, one must first build a map of a* as a function
  of concentration, and stretch factors.  The function bldmap()
  will construct the map.  After the map is available, the a*
  estimates can be computed using extast().
    
Contents:
  bldmap()
  extast()

"""

import flohetc, flotrace

from spivet.pivlib import pivdata
from numpy import *

def bldmap(astbnds,nast,sbnds,ns,saxnc):
    """
    ----
    
    astbnds         2-element array of a* bounds.
    nast            Number of points spanning astbnds.
    sbnds           2-element array specifying semi-axes bounds.
    ns              Number of points spanning sbnds.
    saxnc           3-element array of cell-count for convolution.
    
    ----
    
    Construct the a* = a*(conc,s1,s0) map, where conc is the 
    concentration at the center of an ellipse for a particular value
    of a*, s1, and s0. The parameters s1 and s0 correspond to the middle
    and max semi-axes of the ellipsoid.  The third semi-axis length
    will be computed by enforcing the incompressibility constraint
        s0*s1*s2 = 1
    Note that 
        0 <= conc <= 1
    
    The map is constructed by convolving the heat kernel with the
    initial distribution of concentration.  The parameter saxnc
    specifies the number of cells along each dimensions to use
    during the convolution operation.  A recommended value is
    saxnc = [20,20,20].
    
    Returns a PIVData object containing the map.  The map will
    be stored under Epoch 0 of the PIVData object with the
    variable name ASTMAP.  The map will have dimensions of nast 
    x ns x ns cells.
    """
    print "STARTING: bldmap"

    [map,dc,corigin,ds,sorigin] = \
        flohetc.bldmapcore(astbnds,nast,sbnds,ns,saxnc)

    map = pivdata.cpivvar(map,"ASTMAP","NA")
    
    cellsz = [dc,ds,ds]    
    origin = [corigin,sorigin,sorigin]
    
    mpd = pivdata.PIVData(cellsz,origin,"ASTAR MAP")
    mpd.addVars(0,map)
    
    print " | EXITING: bldmap"
    return mpd


def extast(tavs,conc,astmpd):
    """
    ----
    
    tavs            Time-averaged stretch field.
    conc            Target concentration.
    astmpd          PIVData object containing the a* map.
    
    ----
    
    Extracts the a* estimates from the time-averaged stretch field
    in the tavs variable.  tavs should be a PIVVar from a call
    to floftle.tavscomp().
    
    The target concentration at which the a* values should be
    computed is specified by way of conc.  extast() will then
    use linear interpolation into the a* map to compute the a* value
    corresponding to conc and the time-averaged stretch field.
    
    Returns a PIVVar containing a*.
    """
    # Initialization.
    ncells = tavs.shape[1::]
    tavs   = tavs.reshape([3,tavs[0,...].size]).transpose()

    map     = astmpd[0]['ASTMAP']
    mcellsz = astmpd.cellsz
    morigin = astmpd.origin
    
    mcrd      = empty(tavs.shape)
    mcrd[:,0] = conc
    mcrd[:,1] = tavs[:,1]
    mcrd[:,2] = tavs[:,0] 
    mcrd      = (mcrd -morigin)/mcellsz
    
    # Interpolate the a* values.
    ast = flotrace.svinterp(map,mcrd)
    ast = ast.reshape([1,ncells[0],ncells[1],ncells[2]])

    return pivdata.cpivvar(ast,"AST","NA")
    
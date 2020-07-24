"""
Filename:  procdata.py
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
    Post process PIV results.
"""

from spivet import pivlib
from spivet import flolib
from numpy import *

# >>>>> SETUP <<<<<
pdifname  = "PLUME-3DMF5-WFV.ex2"   # Input file for PIVData object.

velkey = "OFDISP-MEDFLTRD"        # Velocity key for vort, divrg, trace.
tssdiv = 3                          # Time steps per Epoch for tracing.

# Passive tracer setup for mpsvtrace() and rpsvtrace().  mpsvtrace()-specific
# parameters are prefixed with m_, while those limitied to rpsvtrace()
# are prefixed by r_.
m_trace = False                    # Advect passive tracers with mpsvtrace.
m_ntrpc = 40                       # Number of tracers per cell, mpsvtrace.
m_adjid = False                    # Force tracers to source ID.
m_cmthd = "M"                      # Composition method.

r_trace  = False                    # Advect passive tracers with rpsvtrace.
r_epslc  = slice(0,8)               # Epoch slice for processing.
r_csdiv  = (6,3,3)                  # Composition subdivision factor (z,y,x).
r_interp = True                     # Composition interpolation flag.
r_pdofname="HRCOMP-VTL-NTRP.ex2"    # Output file for PIVData object. 

def initcomp( comp, csdiv ):
    # comp will be passed in as all zeros.
    ncells = comp.shape[1:4]
    sf     = 1 #csdiv[1]
    nlyrs  = ncells[1]/sf
    cmul   = 32767/(nlyrs -1)

    comp[0,:,-sf::,:] = 0.
    for i in range(1,nlyrs):
        sndx = (-i -1)*sf
        endx = sndx +sf
        yslc = slice(sndx,endx)

        xslc = slice(0,ncells[2])

        comp[0,:,yslc,xslc] = i*cmul

def initsrc( src, csdiv ):
    # src will be passed in filled with -1.0.
    sf     = 1 #csdiv[1]
    ncells = src.shape
    nlyrs  = ncells[1]/sf
    cmul   = 32767/(nlyrs -1)

    zpo = 0

    src[:,0,...] = 32767
    src[:,-sf::,:] = 0.

    for i in range(1,nlyrs):
        sndx = (-i -1)*sf
        endx = sndx +sf
        yslc = slice(sndx,endx)

        src[:, yslc, 0:3] \
            = src[ :, yslc, -3::] \
            = src[ zpo:(zpo+3), yslc,  :] \
            = src[(ncells[0]-zpo-3):(ncells[0]-zpo), yslc,  :] \
            = i*cmul

def initrbndx( ncells ):
    # Will be passed the number of cells in the velocity array.  Must
    # return a 3x2 integer array specifying the rbndx for rpsvtrace().
    # To use the full dataset, return None.
    rbndx = [ [0,ncells[0]],
              [0,ncells[1]],
              [0,ncells[2]] ]

    return rbndx

# Pathlines setup.
ptrace = False                       # Advect passive tracers for pathlines.
epslc  = slice(0,12)                 # Epoch slice for processing.
ids    = [0,1,2]                     # Integer ID's for pathline groups.
pthofname = "PATHLINES-MID"          # Output file name.    
pthdesc   = "PATHLINES"              # File description.

def inittcrd( id, ncells ):
    # Must return an lx3 array containing initial tracer coordinates,
    # with l equal to the number of tracers desired.
    spc = 1

    """
    # For 2D array.
    ndxmat = indices((ncells[0]/spc,ncells[2]/spc),dtype=float)
    ndxmat = ndxmat*spc
    ntrcrs = ndxmat[0,...].size

    zcrd = ndxmat[0,...].reshape(ntrcrs)
    ycrd = empty(ntrcrs,dtype=float)
    xcrd = ndxmat[1,...].reshape(ntrcrs)
    """
    # For 1D array.
    ztrcrs = ncells[0]/spc
    xtrcrs = 0 #ncells[2]/spc
    ntrcrs = ztrcrs +xtrcrs

    zcrd                = empty(ntrcrs,dtype=float)
    zcrd[0:ztrcrs]      = spc*arange(ztrcrs) 
    zcrd[ztrcrs:ntrcrs] = ncells[0]/2

    xcrd                = empty(ntrcrs,dtype=float)
    xcrd[0:ztrcrs]      = ncells[2]/2
    xcrd[ztrcrs:ntrcrs] = spc*arange(xtrcrs)

    ycrd = empty(ntrcrs,dtype=float)

    # Common.
    if ( id == 1 ):
        ycrd[:] = ncells[1] -1
    if ( id == 2 ):
        ycrd[:] = ncells[1] -5        
    if ( id == 3 ):
        ycrd[:] = ncells[1] -10
    if ( id == 4 ):
        ycrd[:] = ncells[1] -30

    tcrd = array([zcrd,ycrd,xcrd])
    return tcrd.transpose()        

# >>>>> END USER MODIFIABLE CODE <<<<<

pd = pivlib.loadpivdata(pdifname)
pd.addsQA("procdata.py")

# Advect tracers.
if ( m_trace ):
    print "Advecting tracers with mpsvtrace()."
    ncells = pd[0][velkey].shape[1:4]

    icomp = pivlib.PIVVar((1,ncells[0],ncells[1],ncells[2]),
                          name="COMPOSITION",
                          units="NA",
                          vtype=pd[0][velkey].vtype)
    initcomp(icomp,array((1,1,1)))

    src        = empty(ncells,dtype=float)
    src[:,...] = -1.
    initsrc(src,array((1,1,1)))

    flolib.mpsvtrace(pd,
                     velkey,
                     icomp,
                     m_ntrpc,
                     tssdiv,
                     src=src,
                     adjid=m_adjid,
                     cmthd=m_cmthd)

if ( r_trace ):
    print "Advecting tracers with rpsvtrace()."
    ncells = pd[0][velkey].shape[1:4]
    rbndx  = array(initrbndx(ncells))
    rsize  = rbndx[:,1] -rbndx[:,0]
    xrsize = r_csdiv*rsize

    icomp = pivlib.PIVVar((1,xrsize[0],xrsize[1],xrsize[2]),
                          name="COMPOSITION",
                          units="NA",
                          vtype=pd[0][velkey].vtype,
                          dtype=int16)
    initcomp(icomp,r_csdiv)

    src        = empty(xrsize,dtype=int16)
    src[:,...] = -1.
    initsrc(src,r_csdiv)

    tpd = flolib.rpsvtrace(pd[r_epslc],
                           velkey,
                           icomp,
                           tssdiv,
                           r_csdiv,
                           src=src,
                           rbndx=rbndx,
                           interp=r_interp)

    print "Saving composition."
    tpd.save(r_pdofname,4)

# Compute pathlines.
if ( ptrace ):
    print "Computing pathlines."
    ncells = pd[0][velkey].shape[1:4]

    xthlst = []
    for id in ids:
        itcrd = inittcrd(id,ncells)
        thlst = flolib.pthtrace(pd[epslc],velkey,itcrd,tssdiv)

        xthlst.append([id,thlst])

    print "Saving pathlines."
    flolib.pthwrite(xthlst,pthofname,pthdesc)

"""
Filename:  pivpost.py
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
  Module containing post processing routines for data produced
  by PivLIB.

Contents:
  divfltr()
  medfltr()
  gsmooth()
"""

from numpy import *
from spivet.flolib import floutil
from scipy import stats
import pivdata, pivlibc, pivutil
from spivet import compat

#################################################################
#
def divfltr(var,cellsz,divp=99,planar=True,maxits=100):
    """
    ----
    
    var            # PIVVar to be filtered.
    cellsz         # 3 element array specifying the cell sizes [mm].
    planar=True    # Apply 2D filter to each z-plane.    
    divp=99        # Divergence percentile for filtering.
    maxits=1       # Maximum number of iterations. 
    
    ----

    divfltr() corrects a vector field by enforcing a zero
    divergence criteria.  The method utilized here is based on the
    iterative technique of Goodin:1980.
    
    For constant density flows, the continuity equation simplifies to
        del . u = 0
    where del is the gradient operator and u is the velocity vector.
    This zero divergence condition can provide an additional mechanism 
    for detecting and correcting spurious velocity vectors.  
    
    The iterative technique implemented in divfltr() proceeds as
    follows.  The divergence of the vector field is computed and
    the divp percentile divergence value is extracted to form a 
    threshold, dthold.  In the following, let i,j,k represent indices
    along the z,y,x-axes respectively.
        Iterative Loop:
            - All cells with a divergence higher than dthold are
              flagged as having excessive divergence.
            - For flagged cells, a vector adjustment is computed for 
              each vector component of var as
                  wadj = - div/ncmp * cellsz[0]
                  vadj = - div/ncmp * cellsz[1]
                  uadj = - div/ncmp * cellsz[2]
              where w represents the vector component along the z-axis, v
              represents component along the y-axis, u represents the
              component along the x-axis, and div is the divergence for
              the cell.  ncmp, the number of vector components, is set
              to 3 if planar = False, and 2 otherwise.
              
              NOTE: if planar = True, wadj will not be computed and the
              z-component of var will not be updated.
            - Vector values are then updated as
                  w[i+1,j,k] = w[i+1,j,k] +wadj
                  w[i-1,j,k] = w[i-1,j,k] -wadj
                  v[i,j+1,k] = v[i,j+1,k] +vadj
                  v[i,j-1,k] = v[i,j-1,k] -vadj
                  u[i,j,k+1] = u[i,j,k+1] +uadj
                  u[i,j,k-1] = u[i,j,k-1] -uadj 
            - The loop repeats until all cells have divergences below
              dthold or maxits has been reached.

    Given the nature of the algorithm, a cell with an excessive divergence
    actually causes the velocities of its 6 neighbors to be modified.
    So the number of cells filtered will be larger than that specified
    via the divp percentile.  

    The primary purpose of the planar flag is to permit the user to
    analyze a 2D velocity field or to limit divergence updates to the
    y,x-components of the velocity field only.
    
    NOTE: var must be a PIVVar with 3 vector components regardless of
    the planar setting.
                  
    Returns [fvar,fltrd] where
        fvar --- Filtered version of var.  Variable name will be appended
                 with '-DF'.
        fltrd -- Scalar flag indicating which cells have been modified.
    """
    # Initialization.
    if ( planar ):
        divsf = 0.5
        ncs   = 1
    else:
        divsf = 1./3.
        ncs   = 0

    ncells  = array( var.shape[1::] )
    tncells = ncells[0]*ncells[1]*ncells[2]
    
    daflg = zeros(ncells,dtype=bool)
    
    # A 1 cell wide border will be established around the analysis region.
    # Only the analysis region will be checked against dthold, but all 
    # cells in the variable are subject to updating.
    nrcells = ncells -2
    if ( planar ):
        nrcells[0] = ncells[0]
    tnrcells = nrcells[0]*nrcells[1]*nrcells[2] 
    
    ndxmat = indices(nrcells)
    ndxmat = ndxmat.reshape((3,tnrcells))
    if ( planar ):
        ndxmat[1:3,...] = ndxmat[1:3,...] +1
    else:
        ndxmat = ndxmat +1
    
    # Compute dthold.
    dvdy = floutil.d_di(var[1,...],1,cellsz[1])
    dudx = floutil.d_di(var[2,...],2,cellsz[2])
    if ( not planar ):
        dwdz = floutil.d_di(var[0,...],0,cellsz[0])
        div  = dwdz +dvdy +dudx
    else:
        div  = dvdy +dudx     
    
    div  = div[0,ndxmat[0,:],ndxmat[1,:],ndxmat[2,:]]
    div  = div.reshape(tnrcells)
    adiv = abs(div)
    
    dthold = stats.scoreatpercentile(array(adiv),divp)

    # Main loop.
    mvar = var.copy()
    cnt  = 0    
    while (True):
        if ( cnt >= maxits ):
            break
        
        msk = adiv > dthold
        if ( not msk.any() ):
            break

        cnt = cnt +1

        endx = ndxmat[:,msk]   # Cells that have excessive div.
    
        pdiv = -divsf*div[msk]
    
        # Adjust the components.
        for c in range(ncs,3):
            adj = cellsz[c]*pdiv

            pndx = endx[c,:] +1
            mndx = endx[c,:] -1
            
            if ( c == 0 ):
                pslc = [ pndx, endx[1,:], endx[2,:] ]
                mslc = [ mndx, endx[1,:], endx[2,:] ]                
            elif ( c == 1 ):
                pslc = [ endx[0,:], pndx, endx[2,:] ]
                mslc = [ endx[0,:], mndx, endx[2,:] ]
            elif ( c == 2 ):
                pslc = [ endx[0,:], endx[1,:], pndx ]
                mslc = [ endx[0,:], endx[1,:], mndx ]

            mvar[c,pslc[0],pslc[1],pslc[2]] = \
                mvar[c,pslc[0],pslc[1],pslc[2]] +adj

            mvar[c,mslc[0],mslc[1],mslc[2]] = \
                mvar[c,mslc[0],mslc[1],mslc[2]] -adj

            daflg[pslc[0],pslc[1],pslc[2]] = True
            daflg[mslc[0],mslc[1],mslc[2]] = True
                                            
        # Compute updated divergence.
        dvdy = floutil.d_di(mvar[1,...],1,cellsz[1])
        dudx = floutil.d_di(mvar[2,...],2,cellsz[2])
        if ( not planar ):
            dwdz = floutil.d_di(mvar[0,...],0,cellsz[0])
            div  = dwdz +dvdy +dudx
        else:
            div  = dvdy +dudx
            
        div  = div[0,ndxmat[0,:],ndxmat[1,:],ndxmat[2,:]]
        div  = div.reshape(tnrcells)
        adiv = abs(div)
    
    mvar.setAttr("%s-DF" % (var.name),var.units,var.vtype)    
    daflg = pivdata.cpivvar(daflg.reshape(ncells),
                            "%s-DFFLG" % (var.name),"NA",var.vtype)
    
    return [mvar,daflg]


#################################################################
#
def medfltr(var,fdim,rthsf=2.,reps=0.,planar=True,nit=1,cndx=None):
    """
    ----
    
    var            # PIVVar to be filtered.
    fdim           # Filter dimension.
    rthsf=2.       # Residual threshold scale factor.
    reps=0.        # Residual epsilon.
    planar=True    # Apply 2D filter to each z-plane.
    nit=1          # Number of iterations. 
    cndx=None      # Component index to filter.
    
    ----
        
    medfltr() uses a modified form of the universal outlier
    detection scheme of Westerweel (Westerweel:2005).  At each
    data point (i,j,k) in var, a symmetric region of size fdim, 
    where fdim is scalar and odd, is selected and the median of
    all cells in that region including cell (i,j,k) is computed.  
    The residual for the (i,j,k) data point is then computed as

        r_(i,j,k) = | var_(i,j,k) - med_(i,j,k) |

    where var_(i,j,k) is the value of var at (i,j,k), and
    med_(i,j,k) is the median of the region as described above.

    If the residual at a given (i,j,k) exceeds a threshold, then
    the value of var at (i,j,k) is set to the median value of
    var in the region.  The residual threshold, rth, applicable for 
    each region is set to 
        rth = rthsf*( rm +reps ) 
    where rm is the median of the residuals for the region.  Again, 
    the medians here are computed by including values at point 
    (i,j,k).

    A good starting value of reps for application of medfltr() to
    2D optical flow results is the RMS error of displacements (generally
    around 0.1 pixels for PIV data).  Similarly, if medfltr() is 
    applied to temperature results, set reps to the an estimate of
    the uncertainty in extracted temperatures.  fdim should usually be 
    no larger than the largest flow feature of interest.  If fdim is 
    larger than these features, they will tend to be smoothed.  The 
    default value of rthsf, rthsh=2., is a good starting point for all 
    use of medfltr().

    If planar is False, a 3D median filter (fdim x fdim x fdim) will 
    be applied to the dataset.

    medfltr() can be run several times on a single variable.  This permits
    the use of a small fdim, while still providing robust elimination of
    clusters of spurious cells.  The number of times medfltr() is run on
    a variable is specified by nit.

    If cndx = None, all components of the variable will be filtered.
    Otherwise, cndx can be set to the appropriate index of the component
    to filter.

    Returns [fvar,fltrd] where
        fvar --- Filtered version of var.  Variable name will be appended
                 with '-MF'.
        fltrd -- Scalar flag indicating which vector components of 
                 fvar have been modified.  fltrd value will be 
                 set using the following constants
                     Vector        fltrd
                     Component     Value
                       0             1
                       1             2
                       2             4
                  Note that the fltrd value will be scaled by a factor
                  of pow(8,i) where i is the particular iteration.
    """
    # Initialization.
    fdim = int(fdim)
    if ( mod(fdim,2) == 0 ):
        print "ERROR: medfltr dimension must be an odd integer."
        return

    if ( planar ):
        pint = 1
    else:
        pint = 0

    nzelms = var.shape[1]
    nyelms = var.shape[2]
    nxelms = var.shape[3]

    ncmp = var.shape[0]

    if ( compat.checkNone(cndx) ):
        cstrt = 0
        cstop = ncmp
    else:
        cstrt = cndx
        cstop = cndx +1


    fvar = var.copy()
    fvar.setAttr("%s-MF" % var.name,var.units,var.vtype)

    fltrd = pivdata.PIVVar( (1,nzelms,nyelms,nxelms), 
                            "%s-MFFLG" % var.name, "NA", dtype=int )

    if ( ncmp > 3 ):
        print "ERROR: medfltr not supported for PIVVar's with ncmp > 3."
        return

    fltrv = [1,2,4]

    # Filter loop.
    for i in range(nit):
        ffo = pow(8,i)

        for c in range(cstrt,cstop):
            # Compute the median (med), residual (res), and scaled
            # median of the residuals (mres).
            med  = pivlibc.wcxmedian_core(fvar[c,:,:,:],fdim,pint)
            res  = abs( fvar[c,:,:,:] -med )
            mres = rthsf*(pivlibc.wcxmedian_core(res,fdim,pint) +reps)
            
            # Apply the filter.
            msk            = res > mres
            fvar[c,:,:,:]  = where(msk, med, fvar[c,:,:,:])
            fltrd[0,:,:,:] = where(msk, 
                                   fltrd[0,:,:,:] +fltrv[c]*ffo, 
                                   fltrd[0,:,:,:])
            
    return [fvar,fltrd]

#################################################################
#
def gsmooth(var,gbsd,planar=True,nit=1,cndx=None):
    """
    ----
    
    var            # PIVVar to be filtered.
    gbsd           # Gaussian blur standard deviation.
    planar=True    # Apply 2D filter to each z-plane.
    nit=1          # Number of iterations.
    cndx=None      # Component index to filter.
    
    ----
        
    gsmooth() applies a symmetric Gaussian kernel to the PIVVar var.
    The smoothing kernel will have a standard deviation of gbsd along
    two axes if planar is True, and along three axes if planar is False.
    gbsd has units of 'cells' (ie, not mm).
    
    Unlike medfltr(), gsmooth() modifies all values indiscriminately.  
    The name of the filtered variable will be appended with '-GS'.
    
    If planar is False, a 3D Gaussian filter will be applied to the 
    dataset.

    gsmooth() can be run several times on a single variable.  The number
    of times gsmooth() is run on a variable is specified by nit.

    If cndx = None, all components of the variable will be filtered.
    Otherwise, cndx can be set to the appropriate index of the component
    to filter.

    Returns fvar, the filtered variable.
    """
    from scipy import ndimage
    
    # Initialization.
    if ( not isinstance(var, pivdata.PIVVar) ):
        raise TypeError("gsmooth() must be passed a PIVVar.")
    
    if ( planar ):
        # The easiest way to handle planar filter is to run a 3D filter
        # with off-plane entries set to zero.
        pkrnl = pivutil.bldgbkern([gbsd,gbsd])

        kdim  = pkrnl.shape[0]
        krnl  = zeros([kdim,kdim,kdim],dtype=float)

        krnl[(kdim-1)/2,...] = pkrnl        
    else:
        krnl = pivutil.bldgbkern([gbsd,gbsd,gbsd])

    ncmp = var.shape[0]
    
    if ( compat.checkNone(cndx) ):
        cstrt = 0
        cstop = ncmp
    else:
        cstrt = cndx
        cstop = cndx +1

    fvar = var.copy()
    fvar.setAttr("%s-GS" % var.name,var.units,var.vtype)

    # Apply the filter.
    for i in range(nit):
        for c in range(cstrt,cstop):
            fvar[c,...] = ndimage.convolve(fvar[c,...],krnl,mode='nearest') 

    return fvar

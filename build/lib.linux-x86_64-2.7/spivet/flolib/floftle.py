"""
Filename:  floftle.py
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
  Module providing Finite Time Lyapunov Exponent (FTLE) and stretch 
  factor functionality for FloLIB.

  The FTLE field provides a measure of how much particles separate over
  time.  Regions with high FTLE values indicate that nearby particle
  trajectories separate at most exponentially fast, at least for the 
  finite time period for which the FTLE was computed.  Extrema of the 
  FTLE field denote Lagrangian coherent structures (LCS) some of which 
  are approximations to hyperbolic material lines and surfaces (ie, 
  stable and unstable manifolds), while others are merely indicative of 
  high shear regions.  True hyperbolic material lines/surfaces partition 
  the real flow domain into distinct dynamical zones, however the LCS
  approximation is not completely Lagrangian (ie, there is generally a
  small but usually neglegible flux across the LCS).
  
  The computation of the FTLE field follows that of Shadden:2005.  For
  more information on LCS in general, see Haller:2001, Haller:2002,
  and Shadden:2005.  Shadden:2005 provides a detailed look at the flux
  across LCS boundaries.  Haller:2002 (Theorem 3) provides a criteria
  that can be used to distinguish between hyperbolic and parabolic (ie,
  regions of high shear) LCS.

  A generic overview of the FTLE extraction procedure utilized by floftle
  is as follows.  A large number of passive tracers are advected using
  a given velocity field.  The position of the tracers as a function of
  time provides a 'flow map,' a history of a particular tracer's position
  as a function of time.  The finite time right Cauchy-Green strain tensor
  is then formed from the flow map, and the maximum eigenvalue of the 
  tensor is used to compute the FTLE.  Mathematically, the procedure is 
  as follows.  Let phi be the flow map, set
      D = dphi/dx
  where D is in essence the Jacobian of the flow map.  The right 
  Cauchy-Green tensor becomes
      RCG = D'D
  where the prime represents a standard matrix transpose.  The FTLE is
  then
      sigma = (1/T)*ln( sqrt( lamda_max ) )
  where lamda_max is the max eigenvalue of the RCG tensor, and T is the
  time interval over which the flow map was computed.  Again, for more
  of the mathematical underpinnings see any of the above references
  (the notation here is that of Shadden:2005).

  A few particular notes about the FTLE computation algorithm used in
  floftle.  
    1) floftle utilizes the passive tracer advection facilities of
    flotrace.mpsvtrace(), so all of the considerations about memory
    usage, etc described in the documentation for mpsvtrace() apply here
    as well.

    2) Numerical or experimental velocity fields are not only limited to
    availability over finite time intervals, their spatial resolution is
    finite as well.  Consequently, the effect of flow boundaries is only
    captured in a piecewise continuous sense by the velocity field.  The
    end result of this discretization error is that some tracers will see
    velocity vectors that cause them to advect beyond the flow domain.
    These tracers need to be handled specially during FTLE computation
    just before they leave the domain, otherwise the FTLE data they carry
    will be lost.  floftle stops advection of any tracers leaving the
    domain and computes the FTLE for that tracer just prior to it
    leaving.  floftle does not use any special techniques to simulate
    impermeable boundaries of closed flow domains (ie, tracers are
    allowed to leave and the FTLE is computed for those tracers as just
    described).
    
    3) The flow map gradient is in general computed using second order
    central differences when possible.  There are however situations
    where this is not possible (eg, for the tracer launched from a cell
    on the perimeter of the flow domain).  In those cases, the gradient
    computation will fall back to a first order forward or backward
    difference.

    4) floftle permits a fine grid of tracers to be used for FTLE
    computation.  In general, the finer the tracer grid, the more
    spatially resolved FTLE features will become.  However, the user
    should be aware that the spatial resolution of the velocity field
    is a limiting factor to how much FTLE structure can be revealed.

    5) The trilinear spatial and linear temporal interpolation scheme
    of mpsvtrace() applies to FTLE tracers as well.

  Now for an overview of the the stretch field computations performed by
  the flowftle module.  A general flow field will deform an infinitesimal 
  ball into an ellipsoid, with the ratios of ellipsoid principal axis
  lengths to the ball radius denoted as (principal) stretch factors.
  Like the FTLE, these stretch factors provide insight into how 
  nearby particles separate.  The stretch factors are also a diagnostic 
  of how a flow deforms fluid elements and how efficient the flow might 
  be at destroying heterogeneity through diffusive mixing.  The stretch
  factors are simply the singular values of D, the gradient of the flow
  map as defined above.  Consequently, the FTLE and the stretch factors
  are intimately related, with the FTLE being a measure of the time rate
  of change of the stretch factors.  For an incompressible flow, the
  product of the stretch factors must be equal to 1.  This is simply a
  statement of volume conservation: the infinitesimal ball may deform into
  an ellipsoid under the influence of the incompressible flow, but its
  volume must not change.  Unless otherwise noted, the floftle module 
  assumes that all flows are incompressible when computing stretch 
  factors.  The same underlying algorithms used to compute the FTLE are
  used to compute the stretch factors.  
  
  The flowftle module permits the user to compute stretch factors in two 
  ways.  A call to ftlecomp() will return the FTLE field for the time
  of interest along with the corresponding stretch factors.  These
  stretch factors are an instantaneous representation of the accumulated
  deformation that has occurred from time t = 0 up until the Epoch of
  interest.  That is they represent the axis lengths of the deformed
  ellipsoid at the conclusion of tracer advection.  The user can also
  compute a time-averaged representation of the stretch factors via a
  call to tavscomp().  This time-averaged stretch field then represents
  the mean axis lengths of the deformed ellipsoid from t = 0 until
  termination of tracer advection. 

  To utilize the floftle module, first call ftleinit() to initialize
  some parameters that are used throughout FTLE/stretch factor 
  computation.  Next, ftletrace() should be called to advect tracers.  
  The FTLE and time-averaged stretch fields can then be computed via 
  calls to ftlecomp() and/or tavscomp() using the same tracer field.
  
  
Contents:
  class TCAdjuster

  ftlecomp()
  ftleinit()
  ftletrace()
  tavscomp()
"""

import flotrace, floftlec

from spivet.pivlib import pivdata
from numpy import *
from spivet import compat

#################################################################
#
class TCAdjuster:
    """
    Simple class that handles coordinate adjusting and composition
    for ftletrace() during multiple calls to mpsvtrace().
    """
    def __init__(self,ncells,ntrpc,ncalls,srbndx):
        """
        TCAdjuster maintains a counter, m_called, that is 
        incremented each time adjust() is called.  During a call to
        adjust(), the k coordinates for all tracers are incremented 
        as necessary so that the full set of tracers are uniformly 
        distributed throughout a cell.
        
        Note: ncells is with regard to the full velocity mesh ignoring
        srbndx.
        """
        self.m_called = 0
        
        # Compute parameters necessary to shift tracers along the
        # k-axis.
        #
        # anktrpc ----- Aggregate number of tracers per cell along k-axis.
        # m_aorigin --- k-origin for tracer 0.
        # m_atcellsz -- Aggregate cellsz along k-axis. 
        anktrpc = ncalls*ntrpc[0]
        
        self.m_ncells   = array(ncells)
        self.m_aorigin  = -(anktrpc -1.)/(2.*anktrpc)
        self.m_atcellsz = 1./anktrpc
        self.m_ntrpc    = array(ntrpc)
        self.m_ncalls   = ncalls
        self.m_srbndx   = array(srbndx)

    def adjust(self,tcrd,ntrpc,pd):
        tncells = self.m_ncells.prod()
        srbndx  = self.m_srbndx

        # Offset along z-axis for ncalls.
        ccrdz = floor(tcrd[:,0] +0.5)
        dcrdz = tcrd[:,0] -ccrdz

        koset = self.m_called*self.m_ntrpc[0]*self.m_atcellsz +self.m_aorigin
        
        dcrdz     = (dcrdz -dcrdz[0])/float(self.m_ncalls) +koset
        tcrd[:,0] = ccrdz +dcrdz
        
        # Only keep those tracers that are within srbndx.  Tracer 
        # coordinates are cell-centered, so to find the cell faces, we 
        # need to subtract 0.5 from the cell coordinates.
        msk = ones(tcrd.shape[0],dtype=bool)
        for i in xrange(3):
            msk = msk*( tcrd[:,i] >= srbndx[i,0] -0.5 )\
                     *( tcrd[:,i] < srbndx[i,1] -0.5 )
                     
        tcrd = tcrd[msk,:]

        self.m_called = self.m_called +1
        
        return tcrd
    
    def setComp(self,tcrd,pd):
        trid = zeros(tcrd.shape[0],dtype=int16)
        return trid
    
    def reset(self):
        self.m_called = 0
        

#################################################################
#
def ftlecomp(        
    irspath, ftledict     
    ):
    """
    ----

    irspath         # Path to store results from each psvtrace() call.
    ftledict        # Dictionary from ftleinit().

    ----

    ftlecomp() computes the FTLE field using tracers advected with
    ftletrace().  The tracer field from ftletrace() must be stored
    in the path given by irspath.

    For an overview of the FTLE computation procedure, see the
    documentation for the floftle module.
    
    Returns fpd, a PIVData object containing the variables:
        FTLE --------- FTLE field at end of advection.  
        STRETCH ------ Stretch field at end of advection.  The stretch field
                       is stored as a vector with the vector components
                       sorted in order of decreasing magnitude (ie, the
                       maximum stretch factor is given by STRETCH[0,...]).
                       When viewing the stretch field with Paraview, the
                       maximum stretch factor is the 'Z' component of the
                       vector.
                       
                       Note: No attempt is made to enforce the 
                       incompressible condition that the product of stretch
                       factors must equal 1.  Consequently, the smallest
                       stretch factor for a 2D velocity field will be 0.
        MAXST_ORIENT - Unit vector aligned with the direction of maximum stretch
                       at the end of advection.
    The Epochs in fpd will always be arranged in order of increasing time.
    """
    # Initialization.
    print "STARTING: ftlecomp"
    etimes  = ftledict['etimes']
    fepcs   = ftledict['fepcs']
    rxtimes = ftledict['rxtimes']
    ncells  = ftledict['ncells']
    cellsz  = ftledict['cellsz']
    origin  = ftledict['origin']

    tncells = ncells.prod()

    # Setup the new PIVData object to hold the FTLE field.
    fpd = pivdata.PIVData(cellsz,origin)

    # Assemble results and compute FTLE.
    ecnt = 0
    etim = zeros(len(fepcs),dtype=float)
    for e in fepcs:
        print " | Computing FTLE for Epoch %i" % e
        erxtimes   = rxtimes[ecnt]
        etim[ecnt] = etimes[e]

        # t2cmap provides indices into ftle and tcrd0 for each tracer.
        # The c2tmap provides indices into tcrd for each cell (set to -1
        # for cells whose tracer is byp).
        # mshndx = indices(ncells).reshape(3,tncells).transpose()
        t2cndx = arange(tncells)
        c2tndx = arange(tncells).reshape(ncells)

        ftle  = zeros(ncells,dtype=float)
        sfac  = ones([3,ncells[0],ncells[1],ncells[2]],dtype=float)
        mxsfo = zeros([3,ncells[0],ncells[1],ncells[2]],dtype=float)
        
        pth = "%s/EPOCH-%i" % (irspath,e)
        [tscnt,varlst]     = flotrace.mptassy(pth,0)
        [tcrd0,trid,vtmsk] = varlst
        tcrd0 = tcrd0*cellsz  # Ignoring origin.

        vtmsk = vtmsk.squeeze()

        c2tndx = floftlec.bldc2tndx(t2cndx,vtmsk,ncells)
        t2cndx = t2cndx[vtmsk]

        tpc = 0
        for ts in xrange(1,tscnt):
            dt = erxtimes[ts]

            # Load the tracers for the current timestep
            [tscnt,varlst]    = flotrace.mptassy(pth,ts)
            [tcrd,trid,vtmsk] = varlst
            tcrd = tcrd*cellsz  # Ignoring origin.

            vtmsk = vtmsk.squeeze()

            if ( ts == tscnt -1 ):
                vtmsk[:] = False

            # Compute FTLE at the current timestep for those points that
            # will advect beyond the domain in moving to the next
            # timestep.
            floftlec.ftlecore(tcrd0,tcrd,t2cndx,c2tndx,vtmsk,dt,
                              ftle,sfac,mxsfo)

            # Update c2tndx and t2cndx.
            c2tndx = floftlec.bldc2tndx(t2cndx,vtmsk,ncells)
            t2cndx = t2cndx[vtmsk]
            
            if ( mod(ts,tscnt/10) == 0 ):
                tpc = tpc +1
                print " |-| %i%% complete" % (10*tpc)
            
        # Store the fields.
        ftle  = pivdata.cpivvar(ftle,"FTLE","1_S")
        sfac  = pivdata.cpivvar(sfac,"STRETCH","NA")
        mxsfo = pivdata.cpivvar(mxsfo,"MAXST_ORIENT","NA")
        fpd.addVars(ecnt,[ftle,sfac,mxsfo])
        
        ecnt = ecnt +1

    fpd.setTimes(etim)
    if ( fepcs[-1] -fepcs[0] < 0 ):
        fpd = fpd[::-1]
    
    print " | EXITING: ftlecomp"
    return fpd
        
        
#################################################################
#
def ftleinit(pd, epslc, eisz, ntrpc, ncalls, tssdiv, srbndx=None):
    """
    ----

    pd              # PIVData object containing the velocity data.
    epslc           # Epoch slice.
    eisz            # Number of Epochs intervals over which to integrate.
    ntrpc           # Number of tracers per cell along [z,y,x].
    ncalls          # Number of calls to psvtrace().    
    tssdiv          # Number of subtime steps for each Epoch time step.
    srbndx=None     # Region of cells to seed.

    ----
    
    Creates a dictionary containing FTLE related parameters.  ftleinit()
    should be called prior to calling other FTLE functions.
    
    epslc is a python slice object specifying the range of pd Epochs for 
    which tracers should be advected.  For each Epoch in epslc, tracers 
    will be advected for eisz Epoch intervals.  As an example, let
    epslc = slice(24,22,-1) with eisz = 4.  In this case tracers will be
    advected backward twice, once starting at Epoch 24 
        (E24 -> E23, E23 -> E22, E22 -> E21, E21 -> E20),
    and once at Epoch 23 
        (E23 -> E22, E22 -> E21, E21 -> E20, E20 -> E19).
    In each case, tracer coordinates will be computed at 5 Epochs spread
    over 4 Epoch intervals.  Generally speaking, the longer tracers
    are advected, the more LCS structure will be revealed.
    
    ntrpc is a list specifying the number of tracers to be used along each 
    dimension, ncalls specifies the number of times psvtrace() will be
    called.  ncalls is mainly intended to reduce the memory burden of
    tracer advection.  For each call, ntrpc tracers will be advected
    with tracers staggered accordingly along k (the cell-index axis
    parallel to z).  For example ntrpc = [2,4,4] and ncalls = 2 will
    advect 32 tracers per cell for each call.  The first call will
    advect ntrpc oriented at the back half of the cell (along k), and
    the second call will advect ntrpc at the front of the cell (along k).
    Note that ncalls is currently of no benefit for 2D datasets.
    
    tssdiv and irspath are described in the documentation for mpsvtrace().

    In some cases, the FTLE field for a limited region of the flow field 
    is of primary interest.  To compute the FTLE solely for this region,
    one could pass in a truncated version of the velocity field.  The
    disadvantage of this approach, however, is that tracers are likely to
    leave the computational domain of the smaller dataset thereby reducing
    the fidelity of the corresponding FTLE field.  A better approach is to
    use the full velocity field, but reduce (via srbndx) the extent of the 
    region that actually gets seeded with tracers.  This limited subset of 
    tracers is then free to move throughout the full domain of the complete 
    velocity field.  If provided, srbndx must be a 3 x 2 array of cell 
    indices.  Only those cells in the velocity mesh with indices such that
        srbndx[0,0] <= kndx < srbndx[0,1] and
        srbndx[1,0] <= jndx < srbndx[1,1] and
        srbndx[2,0] <= indx < srbndx[2,1]
    will be seeded with ntrpc tracers per cell.  All other cells will be 
    ignored.  If srbndx is None, then all cells will be seeded.  
    
    Returns ftledict, a dictionary containing:
        eisz ------- The eisz parameter.
        epslc ------ Reference to the epslc passed to ftleinit().
        etimes ----- A list of time values for *all* Epochs in pd.
        fepcs ------ List of Epochs for which FTLE should be computed.
                     The set of fepcs are given by epslc.
        ncalls ----- The ncalls parameter.
        ntrpc ------ Reference to the ntrpr passed to ftleinit().
        rxtimes ---- Relative extended time array.  The relative extended
                     times are valid at each subdivided timestep and
                     are relative to TS0 for the current Epoch.  Each
                     Epoch in epslc will have a list of n relative times,
                     where n = eisz*tssdiv.
        tepslc ----- List of slices for tracer advection.
        tssdiv ----- The tssdiv parameter.
        srbndx ----- The srbndx parameter.
        cellsz ----- Cell size for FTLE grid.
        ncells ----- Number of cells in the FTLE grid.
        origin ----- Origin of the FTLE grid.
    """
    # Initialization.
    ftledict = {}
    etimes   = pd.getTimes()
    
    # Build the dictionary.
    ftledict['etimes'] = etimes
    ftledict['epslc']  = epslc
    ftledict['eisz']   = eisz
    ftledict['ntrpc']  = ntrpc
    ftledict['ncalls'] = ncalls
    ftledict['tssdiv'] = tssdiv

    # Check that eisz doesn't extend past available Epochs.
    mxep = len(pd) -1
    mnep = 0
    if ( compat.checkNone(epslc.step) ):
        step = 1
    elif ( abs(epslc.step) > 1 ):
        raise ValueError("|epslc.step| > 1")
    else:
        step = epslc.step

    err = False
    if ( compat.checkNone(epslc.stop) ):
        err = True
    else:
        endep = epslc.stop +step*(eisz -1)
        if ( endep < mnep or endep > mxep ):
            err = True
    if ( err ):
        raise ValueError("Insufficient Epochs for epslc and eisz combination.")

    # Build relative extended time array and slices for tracer advection.
    # The relative extended times are valid at each subdivided timestep
    # (via tssdiv) and are relative to TS0 (ie, they give elapsed time).
    epcs    = arange(len(pd))[epslc]
    rxtimes = []
    tepslc  = []
    for e in epcs:
        testop = e +step*(eisz+1)
        if ( testop < 0 ):
            testop = None
        tepslc.append(slice(e,testop,step))

        erxtimes = []            
        erxtimes.append(0.)
        for ie in xrange(e,e+step*eisz,step):
            edt = etimes[ie+step] -etimes[ie]
            idt = edt/tssdiv
            
            it = idt*arange(1,tssdiv+1,dtype=float) 
            it = it +etimes[ie] -etimes[e]
            
            erxtimes.extend(it)

        rxtimes.append(erxtimes)

    ftledict['tepslc']  = tepslc
    ftledict['rxtimes'] = rxtimes
    ftledict['fepcs']   = epcs

    # Compute FTLE grid parameters so that FTLE will spatially register 
    # with underlying PIVData object containing the velocity field.
    cellsz = array(pd.cellsz)/array(ntrpc)
    if ( compat.checkNone(srbndx) ):
        srbndx      = zeros([3,2],dtype=int)
        srbndx[:,1] = array(pd[0].eshape)
    else:
        srbndx = array(srbndx)

    ncells = (srbndx[:,1] -srbndx[:,0])*array(ntrpc)     
    origin = array(pd.origin) +srbndx[:,0]*pd.cellsz \
            -(array(ntrpc) -1.)*cellsz/2.

    ftledict['cellsz'] = cellsz
    ftledict['ncells'] = ncells
    ftledict['origin'] = origin
    ftledict['srbndx'] = srbndx
    
    return ftledict


#################################################################
#
def ftletrace(
    pd, vkey, irspath, ftledict, interp=['L','L'], hist=2, ndpd=True
    ):
    """
    ----

    pd               # PIVData object containing the velocity data.
    vkey             # Velocity variable name within pd.
    irspath          # Path to store results from each psvtrace() call.
    ftledict         # Dictionary from ftleinit().
    interp=['L','L'] # Interpolation method [Space,Time].
    hist=2           # Tracer history mode.
    ndpd=True        # Non-dimensionalize velocities in pd.

    ----

    ftletrace() performs tracer advection necessary for extraction of
    the FTLE field.  ftletrace() is essentially a wrapper around
    mpsvtrace().  See mpsvtrace() documentation for details on advection
    scheme and the interp parameter.

    NOTE: If tracers are being advected within a plane, be sure to set
    the out of plane velocity component to zero.  Otherwise, tracers are
    prone to leave the domain under consideration.

    Tracers will be stored in a series of ExodusII files organized
    by a simple directory structure rooted at irspath.  Under irspath,
    a single directory will be created for each Epoch in
    ftledict['fepcs'].  Within each of these subdirectories, the tracer
    coordinates at each timestep will be stored in a sequentially
    numbered file.  The file naming convention is that of mpsvtrace().

    NOTE: ftletrace() simply overwrites existing files in irspath.  If
    irspath exists, ftletrace() will not delete the directory and create
    a new one.  This behavior is meant to protect the user from
    accidentally deleting an important directory, but it can cause
    corruption of the FTLE field.  If the user is iteratively running
    flowtrace() and storing the tracer field in the same irspath,
    the existing irspath should be renamed or removed prior to calling
    ftletrace().
    
    The hist parameter controls the granularity at which tracers are
    dumped.  The default value should in general not be changed.
    Nevertheless, hist=2 or hist=3 is supported for tracing (but only
    hist=2 is supported for FTLE extraction!).  See psvtrace()
    documentation for details.

    ftletrace() does not return a value.
    """
    # Initialization.
    fepcs  = ftledict['fepcs']
    ntrpc  = ftledict['ntrpc']
    ncalls = ftledict['ncalls']
    tssdiv = ftledict['tssdiv']
    srbndx = ftledict['srbndx']
    tepslc = ftledict['tepslc']

    if ( hist != 2 and hist !=3 ):
        raise ValueError("hist must be 2 or 3.")

    if ( hist != 2 ):
        print "WARNING: ftletrace() using hist=%i" % hist

    nvcells = array( pd[0].eshape )  # Cells in velocity grid.

    tca = TCAdjuster(nvcells,ntrpc,ncalls,srbndx)

    # If ncalls = 1, get tracer coordinates, otherwise let psvtrace()
    # handle tracer coordinate construction so we can pass the 
    # TCAdjuster.adjust() function.
    if ( ncalls == 1 ):
        tcfun = flotrace.gettcrds(srbndx[:,1] -srbndx[:,0],ntrpc,rsd=None)
        tcfun = tcfun +srbndx[:,0]
    else:
        tcfun = tca.adjust

    # Advect tracers.
    ecnt = 0
    for e in fepcs:
        advslc = tepslc[ecnt]

        tca.reset()

        advirspath = "%s/EPOCH-%i" % (irspath,e)
        flotrace.mpsvtrace(pd,vkey,tca.setComp,ntrpc,ncalls,tssdiv,
                           advirspath,hist=hist,tcfun=tcfun,epslc=advslc,
                           interp=interp,ndpd=ndpd)

        ecnt = ecnt +1
        
        
#################################################################
#
def tavscomp(        
    irspath, ftledict
    ):
    """
    ----

    irspath         # Path to store results from each psvtrace() call.
    ftledict        # Dictionary from ftleinit().

    ----

    tavscomp() computes the time-averaged stretch field using tracers 
    advected with ftletrace().  The procedure assumes all flows are 3D
    works as follows. 
      Naturally 3D flow field:
        For each tracer, the ratios beta and gamma are time-averaged, where
            beta  = s[0]/s[2]
            gamma = s[1]/s[2]
            s[0] >= s[1] >= s[2]
        and the stretch field at time t is given by s.  After time-averaging
        beta and gamma, the corresponding average stretch factors are then
        computed as
            s_ave[2] = ( 1/(beta_ave*gamma_ave) )**(1/3)
            s_ave[1] = s_ave[2]*gamma_ave
            s_ave[0] = s_ave[2]*beta_ave
      2D flow field:
        The assumption that all flow fields are 3D means that the gradient
        of the flow map is singular.  Consequently the smallest stretch 
        factor, which corresponds to the spatial direction that has been
        ignored in this flow, is zero. tavscomp() discards this zero stretch
        factor and replaces it with
            s[2] = 1./(s[0]*s[1])
        The method then proceeds as for the 3D field.
      
    The motivation for time-averaging beta and gamma instead of the stretch
    factors themselves is that by averaging the ratios of the stretch
    factors, we can preserve the volume of the strained ellipsoid.  For 
    incompressible flows, the product of the stretch factors must be 1
        s[0]*s[1]*s[2] = 1
    in order for the volume of a fluid element to be preserved.  Averaging
    the individual stretch factors, as opposed to their ratios, results in 
    a deformed ellipse that does not conserve volume.
    
    The tracer field from ftletrace() must be stored in the path given by 
    irspath.

    NOTE:  Extracting the time-averaged stretch factors is a computationally
    expensive operation.  
    
    NOTE:  tavscomp() enforces the incompressibility constraint that the
    product of the stretch factors must equal 1.
    
    Returns spd, a PIVData object containing the variables:
        TAVE_STRETCH - Stretch field time-averaged over advection duration
                       and sorted such that s[0] >= s[1] >= s[2]
        INTGR_TIME --- Total integration (ie, advection) time used to average
                       the stretch factors.  
    The Epochs in spd will always be arranged in order of increasing time.
    """
    # Initialization.
    print "STARTING: tavscomp"
    etimes  = ftledict['etimes']
    fepcs   = ftledict['fepcs']
    rxtimes = ftledict['rxtimes']
    ncells  = ftledict['ncells']
    cellsz  = ftledict['cellsz']
    origin  = ftledict['origin']

    tncells = ncells.prod()
    
    # Setup the new PIVData object to hold the FTLE field.
    spd = pivdata.PIVData(cellsz,origin)

    # Assemble results and compute FTLE.
    ecnt = 0
    etim = zeros(len(fepcs),dtype=float)
    for e in fepcs:
        print " | Computing average stretch factors for Epoch %i" % e
        erxtimes   = rxtimes[ecnt]
        etim[ecnt] = etimes[e]

        # t2cmap provides indices into ftle and tcrd0 for each tracer.
        # The c2tmap provides indices into tcrd for each cell (set to -1
        # for cells whose tracer is byp).
        # mshndx = indices(ncells).reshape(3,tncells).transpose()
        t2cndx = arange(tncells)
        c2tndx = arange(tncells).reshape(ncells)

        tavsfac = zeros([3,ncells[0],ncells[1],ncells[2]],dtype=float)
        itime   = zeros(ncells,dtype=float)

        pth = "%s/EPOCH-%i" % (irspath,e)
        [tscnt,varlst]     = flotrace.mptassy(pth,0)
        [tcrd0,trid,vtmsk] = varlst
        tcrd0 = tcrd0*cellsz  # Ignoring origin.

        vtmsk = vtmsk.squeeze()

        c2tndx = floftlec.bldc2tndx(t2cndx,vtmsk,ncells)
        t2cndx = t2cndx[vtmsk]

        tpc = 0
        for ts in xrange(1,tscnt):
            dt  = erxtimes[ts]
            idt = dt -erxtimes[ts-1]

            # Load the tracers for the current timestep
            [tscnt,varlst]    = flotrace.mptassy(pth,ts)
            [tcrd,trid,vtmsk] = varlst
            tcrd = tcrd*cellsz  # Ignoring origin.

            vtmsk = vtmsk.squeeze()

            if ( ts == tscnt -1 ):
                vtmsk[:] = False

            # Compute time-weighted stretch factors for the current timestep.
            floftlec.tavscore(tcrd0,tcrd,t2cndx,c2tndx,vtmsk,idt,
                              tavsfac,itime)

            # Update c2tndx and t2cndx.
            c2tndx = floftlec.bldc2tndx(t2cndx,vtmsk,ncells)
            t2cndx = t2cndx[vtmsk]
            
            if ( mod(ts,tscnt/10) == 0 ):
                tpc = tpc +1
                print " |-| %i%% complete" % (10*tpc)

        # Compute the time-averaged stretch factor ratios and stretch 
        # factors.  
        for i in xrange(2):
            tavsfac[i,...] = tavsfac[i,...]/itime

        tavsfac[2,...] = (tavsfac[0,...]*tavsfac[1,...])**(-1./3.)
        
        for i in xrange(2):
            tavsfac[i,...] = tavsfac[i,...]*tavsfac[2,...]

        # Store the fields.
        tavsfac = pivdata.cpivvar(tavsfac,"TAVE_STRETCH","NA")
        itime   = pivdata.cpivvar(itime,"INTGR_TIME","S")
        spd.addVars(ecnt,[tavsfac,itime])
        
        ecnt = ecnt +1

    spd.setTimes(etim)
    if ( fepcs[-1] -fepcs[0] < 0 ):
        spd = spd[::-1]
    
    print " | EXITING: tavscomp"
    return spd
  

"""
Filename:  flotrace.py
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
  Module providing passive tracer functionality for FloLIB.
  
  Passive tracers provide a mechanism for visualizing flow 
  structures and tracking compositional sources within the
  flow field.  In essence, the passive tracers are digital
  fluorescein.  

  The primary driver function within the module is psvtrace().    

Contents:
  bldsvcicoeff()
  gettcrds()
  initfargs()
  mpsvtrace()
  mptassmblr()
  mptassy()
  mptparsefn()
  mptwrite()
  ndvel()
  psvtrace()
  pthtrace()
  pthwrite()
  rpsvtrace()
  _splev()
  _splrep()
  svcinterp()
  svinterp()
  tcseval()
  tcsvar()
  velfun()

  class MPTWriter
"""

import flotracec, floutil, tempfile, os

from spivet.pivlib import pivdata
from spivet.pivlib import exodusII as ex2
from numpy import *

from scipy import interpolate

from spivet import compat
"""
NOTE: Interpolation in all routines is conducted using tracer coordinates
referenced to cell centers.  That is a tracer with coordinates (0,0,0) is
located perfectly in the center of the 0 cell.  The minimum valid coordinate
is then (-0.5,-0.5,-0.5).
"""

class MPTWriter:
    """
    Worker class that handles tracer coordinate writing for
    mpsvtrace().
    """
    def __init__(self,bofpath):
        """
        ----

        bofpath    # Base path in which to store vars.
        
        ----
        
        
        """
        if ( not os.path.exists(bofpath) ):
            os.makedirs(bofpath)

        self.m_call    = 0
        self.m_ts      = 0
        self.m_bofpath = bofpath
        self.m_fnfmt  = "MPSVTRACE-DATA-CALL%i-TS%i"
        

    def setCall(self,call):
        """
        Set call value.
        """
        self.m_call = call

    def setTS(self,ts):
        """
        Set timestep counter.
        """
        self.m_ts = ts

    def incrementTS(self):
        """
        Increment timestep counter.
        """
        self.m_ts = self.m_ts +1

    def write(self,vars,epoch,ts):
        """
        Write the data in vars.  Ignore epoch and ts.
        """
        fname = self.m_fnfmt % (self.m_call,self.m_ts)
        mptwrite(vars,fname,self.m_bofpath)

        # Increment the TS counter.
        self.incrementTS()


#################################################################
#
def bldsvcicoeff(var):
    """
    ----

    var            # Variable to be interpolated.

    ----

    Constructs the polynomial coefficients for Lekien's tricubic
    interpolator (Lekien:2005).

    var must be normalized to the unit cube.

    Returns, coeff, the coefficient matrix that can then be passed
    to flotracec.svcinterp().
    """
    # Initialization.
    vshape = var.shape

    if ( len(vshape) == 3 ):
        var    = var.reshape((1,vshape[0],vshape[1],vshape[2]))
        vshape = var.shape

    vshape = array(vshape)
    ncmp   = vshape[0]

    # If any index has less than two cells, then repeat along that
    # axis so that problem is well posed.  This is equivalent to edge
    # extension.
    #
    # Note that bvmat is one cell smaller than var in each direction.
    # This is due to the way in which shifts are handled in the
    # interpolation routine: namely, the reference cell coordinates
    # against which shifts are computed is determined by flooring the
    # the interpolation coordinates.  For example, an interp coordinate
    # of ( 1.02, 2.56, 3.99 ) is relative to cell ( 1, 2, 3 ).
    if ( (vshape[1::] == 1).any() ):
        for i in range(1,4):
            if ( vshape[i] == 1 ):
                var = var.repeat(2,i)

    vshape = var.shape

    bvmat = empty([ vshape[0],
                    vshape[1] -1,
                    vshape[2] -1,
                    vshape[3] -1,
                    64 ],
                  dtype=float)

    bvshape = bvmat.shape
    npoly   = bvshape[0]*bvshape[1]*bvshape[2]*bvshape[3]

    # Build the bvmat.  This is equivalent to a series of Lekien:2005's
    # b vectors.
    for c in range(ncmp):
        vcmp = var[c,...]

        dfdz = floutil.d_di(vcmp,0,1.)
        dfdy = floutil.d_di(vcmp,1,1.)
        dfdx = floutil.d_di(vcmp,2,1.)

        d2fdzdy = floutil.d_di(dfdz,1,1.)
        d2fdzdx = floutil.d_di(dfdz,2,1.)
        d2fdydx = floutil.d_di(dfdy,2,1.)

        d3fdzdydx = floutil.d_di(d2fdzdy,2,1.)

        vcmp = vcmp.reshape([1,vshape[1],vshape[2],vshape[3]])

        bcl = [vcmp,dfdz,dfdy,dfdx,d2fdzdy,d2fdzdx,d2fdydx,d3fdzdydx]

        cnt = 0
        for j in range( 8 ):
            bcmp = bcl[j]

            bvmat[c,:,:,:,cnt]   = bcmp[0, 0:-1, 0:-1, 0:-1]
            bvmat[c,:,:,:,cnt+1] = bcmp[0, 0:-1, 0:-1, 1:: ]
            bvmat[c,:,:,:,cnt+2] = bcmp[0, 0:-1, 1::,  1:: ]
            bvmat[c,:,:,:,cnt+3] = bcmp[0, 0:-1, 1::,  0:-1]
            bvmat[c,:,:,:,cnt+4] = bcmp[0, 1::,  0:-1, 0:-1]
            bvmat[c,:,:,:,cnt+5] = bcmp[0, 1::,  0:-1, 1:: ]
            bvmat[c,:,:,:,cnt+6] = bcmp[0, 1::,  1::,  1:: ]
            bvmat[c,:,:,:,cnt+7] = bcmp[0, 1::,  1::,  0:-1]

            cnt = cnt +8

    # Compute the polynomial coefficients.
    smat  = flotracec.svcismat()
    bvmat = bvmat.reshape((npoly,64)).transpose()

    coeff = dot(smat,bvmat)
    coeff = coeff.transpose().reshape(bvshape)

    return coeff


#################################################################
#
def gettcrds(ncells,csdiv,rsd=None):
    """
    ----
    
    ncells         # List specifying [z,y,x] cell count.
    csdiv          # Cell subdivision factor.
    rsd=None       # Random seed.
    
    ----
    
    gettcrds() computes initial tracer coordinates based on the
    number of cells in the velocity array and csdiv.  csdiv
    can be a single value or a list specifying the number of
    tracers along each cell axis.  If csdiv is a list, each cell
    will be divided uniformly.  The returned tracer coordinates
    will be referenced to the coarse velocity cells on a 
    cell-centered basis (ie, the minimum valid tracer coordinate
    will be (-0.5,-0.5,-0.5)).

    If rsd is set to an integer, then the tracer coordinates
    will be randomized using rsd as a seed to the random number
    generator.
    
    Returns tcrd, the tracer coordinates.  tcrd is an nx3 array,
    where n is the total number of tracers.
    """
    # Initialization.
    ncells = array(ncells)
    csdiv  = array(csdiv)
    if ( csdiv.size > 1 ):
        ncells = csdiv*ncells
        ntrpc  = 1
    else:
        ntrpc = csdiv
        csdiv = array([1,1,1])
           
    tncells = ncells.prod()
    
    # Get coordinates.
    tcrd = indices(ncells,dtype=float).reshape(3,tncells)
    tcrd = tcrd.transpose()

    if ( ntrpc == 1 ):
        for i in xrange(3):
            tcrd[:,i] = tcrd[:,i]/csdiv[i] -(csdiv[i] -1.)/(2.*csdiv[i])

    if ( ntrpc > 1 ):
        tcrd = tcrd.repeat(ntrpc,axis=0)

    if ( not compat.checkNone(rsd) ):
        for i in xrange(3):
            random.seed(rsd+i)
            tcrd[:,i] = tcrd[:,i] +random.rand(tcrd.shape[0]) -0.5
            
        random.seed(None)

    return tcrd


#################################################################
#
def initfargs(pd,ndvpd,ndvkey,interp):
    """
    ----
    
    pd              # PIVData object containing the velocity data.
    ndvpd           # PIVData object containing non-dimensionalized velocity.
    ndvkey          # Velocity variable name within ndvpd.
    interp          # List specifying interpolation method [space,time].
    
    ----
    
    Simple helper function for passive tracing routines.  initfargs()
    initializes the fargs list that gets passed to the integration
    function and sets the necessary flags for the user-specified
    interpolation scheme.

    ndvpd should be the non-dimensional velocity PIVData object returned 
    from a call to ndvel().

    interp should be a 2-element list, the first item specifying spatial
    interpolation scheme, and the second specifying temporal interpolation
    scheme.  Each item can be either of two values:
        'L' ---- Use linear interpolation (trilinear for space, linear
                 for time).
        'C' ---- Use cubic interpolation (tricubic for space, cubic spline
                 for time).

    Returns fargs, the initialized parameter list.
    """
    fargs = range(7)
    if ( interp[0].upper() == 'L' ):
        fargs[6] = False
    elif ( interp[0].upper() == 'C' ):
        fargs[6] = True
    else:
        raise ValueError("Unsupported: interp[0] = %s" % interp[0])

    if ( interp[1].upper() == 'L' ):
        fargs[5] = None
    elif ( interp[1].upper() == 'C' ):
        tvec  = pd.getTimes()
        tcspl = tcsvar(ndvpd,tvec,ndvkey)

        fargs[5] = tcspl
    else:
        raise ValueError("Unsupported: interp[1] = %s" % interp[1])

    return fargs


#################################################################
#
def mpsvtrace(
    pd, vkey, icomp, ntrpc, ncalls, tssdiv, irspath, src=None, 
    adjid=False, hist=0, ctcsp=True, tcfun=None, epslc=None,
    interp=['L','L'], ndpd=True
    ):
    """
    ----

    pd               # PIVData object containing the velocity data.
    vkey             # Velocity variable name within pd.
    icomp            # Initial composition PIVVar or callback function.
    ntrpc            # Number of tracers per cell per call.
    ncalls           # Number of calls to psvtrace()
    tssdiv           # Number of subtime steps for each Epoch time step.
    irspath          # Path to store results from each psvtrace() call.
    src=None         # (nz,ny,nz) array specifying source cells.
    adjid=False      # Flag to force tracers to source ID in source cells.
    hist=0           # Flag to return tracer coordinates for each step.
    ctcsp=True       # Coerce tracer coordinates to single precision.
    tcfun=None       # Tracer coordinate adjustment function.
    epslc=None       # Epoch slice.
    interp=['L','L'] # Interpolation method [Space,Time].
    ndpd=True        # Non-dimensionalize velocities in pd.
    
    ----

    mpsvtrace() is a wrapper around psvtrace() that minimizes the
    memory footprint during tracer advection.  Generally speaking,
    tracer advection is very memory intensive as a large number
    of particles is being followed through time.  During the process
    of tracer advection, several intermediate arrays, having the
    shape of either the underlying grid or the number of tracers,
    must be created and maintained.  The end result is that tracer
    advection using large ntrpc values can easily exhaust the available
    memory on many machines.

    To overcome the memory exhaustion issue, mpsvtrace() calls
    psvtrace() ncalls times, with ntrpc particles being traced each 
    call.  The total number of tracers advected per psvtrace() call should 
    be kept less than a couple million.  If the hard drive seems to churn 
    excessively during execution, a sign that excessive swapping 
    between RAM and virtual memory is occurring, reduce ntrpc and try 
    again.  Similarly, if malloc errors are generated by the underlying
    OS, then the process has exhausted the available address space.  
    Reduce ntrpc and try again.

    icomp, hist, ctcsp, tcfun, epslc, interp, and ndpd will be passed 
    directly to psvtrace().  ctcsp will force tracer coordinates to be 
    stored in single precision following advection.  hist indicates how 
    frequently results should be returned.  tcfun can be set to a callback 
    function that is used to adjust tracer coordinates prior to advection.  
    epslc can be set to a python slice to indicate which epochs should be 
    used for tracing.  icomp can be set to a function in a manner similar 
    to tcfun.  interp controls the spatial and temporal velocity
    interpolation scheme.  If ndpd is True, the velocity field in pd will
    be non-dimensionalized.  See psvtrace() documentation for more details.
    
    irspath must be set to a writable directory which will be used to
    store the interim results.  If the directory specified by irspath
    does not exist, mpsvtrace() will attempt to create it.  The 
    intermediate results from each call to psvtrace() will be stored in 
    ExodusII files within the specified directory.  Once these files are 
    created, they will not be deleted regardless of downstream errors.  
    ExodusII files created by mpsvtrace() will be named using the 
    following template
        MPSVTRACE-CALL*-TS#
    where * represents the call number to psvtrace(), and # indicates a
    timestep number.  Both * and # start at 0 and are incremented by 1.
    If epslc is specified, TS0 will correspond to the first timestep
    processed regardless of direction (ie, whether reverse tracing from 
    Epoch 7 to 0 or forward tracing from Epoch 7 to 14, the coordinates 
    at Epoch 7 will be written to TS0).  Each file will contain the 
    variables returned by psvtrace().  See the documentation for 
    psvtrace() hist parameter to determine at which timesteps variables 
    are stored.  The list of files within irspath can then be passed to 
    mptassy().  See mptassy() for more details.  

    For more on the tracer advection algorithm, see psvtrace().

    NOTE: Tracer coordinates are normalized by cellsz (ie, they are
    in grid units, not mm).  A coordinate of (0,0,0) is located perfectly 
    in the center of the (0,0,0) cell.

    mpsvtrace() does not return a value.
    """
    print "STARTING: mpsvtrace"    
    # Initialization.
    mptw = MPTWriter(irspath)

    # Main loop.
    for i in range(ncalls):
        print " | ----- GROUP %i -----" % i
        mptw.setCall(i)
        mptw.setTS(0)

        # Advect tracers.
        trdata = psvtrace(pd,vkey,icomp,ntrpc,tssdiv,src=src,
                          adjid=adjid,hist=hist,rsd=i,ctcsp=ctcsp,
                          tcfun=tcfun,epslc=epslc,writer=mptw.write,
                          interp=interp,ndpd=ndpd)

    print " | EXITING: mpsvtrace"
        

#################################################################
#
def mptassmblr(ofplst, rbndx=None):
    """
    ----
    
    ofplst         # List of output file paths.
    rbndx=None     # Region bounding indices.
    
    ----
    
    Helper function for mpsvtrace() that assembles results stored
    in individual files and returns a list of aggregate arrays.  The
    first entry in the returned list will be the tracer coordinate array.
    The other entries will be variables ordered as they were stored (ie,
    the order in which they were returned from psvtrace()).
    
    mptassmblr() makes no assumptions regarding the files in ofplst.  All
    files specified will be combined, with the contents of one file
    appended to the contents of all preceding files.  If only a certain 
    subset of the files should be combined, then only those file paths 
    should be passed in ofplst.

    Each returned variable will be a 2D array of shape n x c, where n
    is the total number of tracers found across all files in ofplst, and
    c is the number of components for that variable (eg, each tracer has
    3 coordinates, so c is 3 for the tracer coordinate array).

    If rbndx is specified, then the returned arrays will only contain
    results for tracers that fall within the cells (or fraction of cells)
    for the region.  rbndx contains the floating point indices of the 
    region such that:
        rbndx[0,0] <= tcrd[:,0] (ie, z-coord) < rbndx[0,1]
        rbndx[1,0] <= tcrd[:,1] (ie, y-coord) < rbndx[1,1]
        rbndx[2,0] <= tcrd[:,2] (ie, x-coord) < rbndx[2,1]
    Note that rbndx should be specified in terms of cell-centered
    coordinates (ie, a coordinate of [0,0,0] is located in the center of
    cell [0,0,0]).  The minimum valid tracer coordinate for the full
    domain is [-0.5,-0.5,-0.5].  The primary benefit of rbndx is to 
    permit analysis of tracer data data for limited regions of interest.
    While the user could potentially filter the full set of tracer coords
    in a similar manner following a call to mptassmblr(), by specifying
    rbndx up front, the coordinates can be filtered as each file in
    ofplst is read.  This greatly reduces the overall memory footprint 
    of tracer data.
    
    Returns varlst, the list of variables.
    """
    # Load the data.
    varlst = [[]]   # Outer is for variable, inner is for file.
    mtntr  = 0      # Masked number of tracers.
    for ofp in ofplst:
        exo = ex2.ex_open(ofp,ex2.exc["EX_READ"],8)
        
        # Tracer coordinates.
        tcrd = ex2.ex_get_coord(exo)
        tcrd = array(tcrd[::-1]).transpose()

        msk = ones(tcrd.shape[0],dtype=bool)
        if ( not compat.checkNone(rbndx) ):
            rbndx = array(rbndx)
            
            for c in range(3):
                msk = msk*( tcrd[:,c] >= rbndx[c,0] )\
                         *( tcrd[:,c] <  rbndx[c,1] )
            
        tcrd = tcrd[msk,:]
        varlst[0].append(tcrd)
        mtntr = mtntr +tcrd.shape[0]
            
        # Other variables.
        info   = ex2.ex_get_info(exo)
        varnms = ex2.ex_get_var_names(exo,"N")

        vcnt    = 1
        ex2vcnt = 0
        for v in varnms:
            if ( vcnt >= len(varlst) ):
                varlst.append([])
            
            [varnm,cmp] = v.rsplit("_",1)
            
            dtype = pivdata.getdtype("%s_" % varnm,info)
            
            cmp = cmp.strip()
            if ( len(cmp) == 0 ):
                ncmp = 1
            elif ( len(cmp) == 1 ):
                ncmp = 3
            elif ( len(cmp) == 2 ):
                ncmp = 9           
    
            ex2vndx = arange(ncmp,dtype=int) +ex2vcnt +1
            ex2vndx = ex2vndx[::-1]
        
            var = []
            for c in range(ncmp):
                var.append( ex2.ex_get_nodal_var(exo,1,ex2vndx[c]) )
            
            var = array(var).transpose().astype(dtype)
            varlst[vcnt].append(var[msk,:])
            
            vcnt    = vcnt +1
            ex2vcnt = ex2vcnt +ncmp
            
        ex2.ex_close(exo)

    # Assemble the data.
    for v in range( len(varlst) ):
        ncmp  = varlst[v][0].shape[1]
        dtype = varlst[v][0].dtype 

        var = empty((mtntr,ncmp),dtype)

        bndx = 0
        for i in range( len(varlst[v] ) ):
            pvar             = varlst[v].pop(0)
            endx             = bndx +pvar.shape[0]
            var[bndx:endx,:] = pvar
            bndx             = endx
            
        varlst[v] = var
        
    return varlst


#################################################################
#
def mptassy(irspath, ts, rbndx=None):
    """
    ----
    
    irspath        # Path where files to be assembled are stored.
    ts             # Time step.
    rbndx=None     # Region bounding indices.
    
    ----
    
    Helper function for mpsvtrace() that assembles results stored
    in individual files and returns a list of aggregate arrays.  
    mptassy() is a wrapper around mptassmblr() that assembles files
    from all psvtrace calls for a given time step (ts).  For the
    timestep specified, file contents will be assembled in increasing 
    order of call number.  
    
    mptassy() returns a list of assembled variables.  The first entry in 
    the returned list will be the tracer coordinate array.  The other 
    entries will be variables ordered as they were stored (ie, the order 
    in which they were returned from psvtrace()).  Each variable will
    be a 2D array of shape n x c, where n is the total number of tracers
    found across all calls for the specified ts, and c is the number
    of components for that variable (eg, each tracer has 3 coordinates,
    so c is 3 for the tracer coordinate array).
    
    If rbndx is specified, then the returned arrays will only contain
    results for tracers that fall within the cells (or fraction of cells)
    for the region.  rbndx contains the floating point indices of the 
    region such that:
        rbndx[0,0] <= tcrd[:,0] (ie, z-coord) < rbndx[0,1]
        rbndx[1,0] <= tcrd[:,1] (ie, y-coord) < rbndx[1,1]
        rbndx[2,0] <= tcrd[:,2] (ie, x-coord) < rbndx[2,1]
    Note that rbndx should be specified in terms of cell-centered
    coordinates (ie, a coordinate of [0,0,0] is located in the center of
    cell [0,0,0]).  The minimum valid tracer coordinate for the full
    domain is [-0.5,-0.5,-0.5].  The primary benefit of rbndx is to 
    permit analysis of tracer data data for limited regions of interest.
    While the user could potentially filter the full set of tracer coords
    in a similar manner following a call to mptassmblr(), by specifying
    rbndx up front, the coordinates can be filtered as each file in
    ofplst is read.  This greatly reduces the overall memory footprint 
    of tracer data.
    
    Returns [tscnt,varlst], where tscnt is the total number of timesteps
    found in irspath, and varlst is the list of variables described above.
    """
    # Initialization.
    dl   = os.listdir(irspath)

    # Get list of all files.
    tscnt   = 0
    callcnt = 0
    tflist  = []
    for f in xrange(len(dl)):
        if ( not dl[f].startswith("MPSVTRACE") ):
            continue

        fnc = mptparsefn(dl[f])
        if ( fnc['TS'] > tscnt ):
            tscnt = fnc['TS']
        
        if ( fnc['TS'] == ts ):
            tflist.append(dl[f])
        
        if ( fnc['CALL'] > callcnt ):
            callcnt = fnc['CALL']

    tscnt   = tscnt +1
    callcnt = callcnt +1

    if ( len(tflist) == 0 ):
        raise ValueError("No files for TS%i found." % ts)
        
    flist = range(callcnt)
    for f in xrange(callcnt):
        fnc  = mptparsefn(tflist[f])
        call = fnc['CALL']
        flist[call] = "%s/%s" % (irspath,tflist[f])

    # Call mptassmblr().
    varlst = mptassmblr(flist,rbndx)
        
    return [tscnt,varlst]


#################################################################
#
def mptparsefn(fn):
    """
    ----
    
    fn             # Filename to parse.
    
    ----
    
    Parses a filename according to mpsvtrace naming convention.
    
    NOTE: mptparsefn() expects fn to be a valid mpsvtrace() file
    name. 
    
    Returns a dictionary containing:
        "CALL" ---- Call number (from mpsvtrace() ncalls).
        "TS" ------ Timestep number.
    """
    fn  = fn[0:-4]
    fnc = fn.rsplit("-",2)

    fnp = {"CALL":int( fnc[1].strip("CALL") ),
           "TS":int( fnc[2].strip("TS") )}

    return fnp


#################################################################
#
def mptwrite(vars,fname,bofpath=None):
    """
    ----
    
    vars           # List of variables to store.
    fname          # File name for output file.
    bofpath        # Base path in which to store vars.
    
    ----
    
    Helper function for mpsvtrace() that stores a group of arrays
    to an ExodusII file.  The first var in vars must be the
    tracer coordinates.  All other floating point data will be stored
    using the dtype of the tracer coordinates.  Additional variables
    are stored using the dtype of the variable.

    The tracer coordinate variable in vars must be a 2D array of
    shape n x 3, where n is the number of tracers with each tracer
    having 3 coordinates ([z,y,x]).  All other variables can be
    1D arrays of length n, or 2D arrays of shape n x c where c is
    the number of components for the variable.  At present, variables
    can have either 1, 3, or 9 components.

    fname provides the output file name.  Do not specify the ex2
    extension.
    
    If bofpath = None, the file will be stored in the standard
    location for temporary files.  Otherwise, the file will stored
    in the directory specified by bofpath.  If bofpath does not
    exist, it will be created.
    
    Returns ofpath, the fully qualified path to the output file.
    """
    # Initialization.
    if ( not compat.checkNone(bofpath) ):
        if ( not os.path.exists(bofpath) ):
            os.makedirs(bofpath)
        
        ofpath = "%s/%s.ex2" % (bofpath,fname)
    else:
        tfh    = temfile.mkstemp()
        os.close(tfh[0])
        ofpath = tfh[1]

    if ( vars[0].dtype == float ):
        fprc = 8
    else:
        fprc = 4
    
    nnodes = vars[0].shape[0]
    
    # Write the data.  Multi-component data should be ordered the same
    # as tcrd.
    exo = ex2.ex_create(ofpath,ex2.exc['EX_CLOBBER'],fprc,fprc)
    ex2.ex_put_init(exo,"MPSVTRACE FILE",3,nnodes,nnodes,1,0,0)
    
    infostr   = []
    ex2varnms = []
    varcmp    = []
    for v in range ( len(vars) ):
        var  = vars[v]
        if ( len( var.shape ) == 1 ):
            ncmp = 1
        else:
            ncmp = var.shape[1] 
        
        varnm = "VAR%i_" % v
        
        infostr.append("%s.DTYPE:%s" % (varnm,var.dtype.name) )
        varcmp.append(ncmp)

        if ( ncmp == 1 ):
            ex2varnms.append(varnm)
        elif ( ncmp == 3 ):
            # ParaView's vector glomming currently requires vector
            # components to be written X, Y, Z.
            ex2varnms.append(varnm +" X")
            ex2varnms.append(varnm +" Y")
            ex2varnms.append(varnm +" Z")
        elif ( ncmp == 9 ):
            ex2varnms.append(varnm +" XX")
            ex2varnms.append(varnm +" XY")
            ex2varnms.append(varnm +" XZ")
            ex2varnms.append(varnm +" YX")
            ex2varnms.append(varnm +" YY")
            ex2varnms.append(varnm +" YZ")
            ex2varnms.append(varnm +" ZX")
            ex2varnms.append(varnm +" ZY")
            ex2varnms.append(varnm +" ZZ")
        else:
            raise ValueError("Unsupported number of components: %i" % ncmp)

    ex2nvars = len( ex2varnms ) -3  # Exclude tracer coordinates.

    ex2.ex_put_info(exo,len(infostr),infostr)

    ex2.ex_put_var_param(exo,"N",ex2nvars)
    ex2.ex_put_var_names(exo,"N",ex2nvars,ex2varnms[3::])
    
    ex2.ex_put_time(exo,1,0)
    
    ex2.ex_put_elem_block(exo,1,"SPHERE",nnodes,1,0)
    ex2.ex_put_elem_conn(exo,1,arange(nnodes)+1)
    
    ex2.ex_put_coord(exo,vars[0][:,2],vars[0][:,1],vars[0][:,0])
    ex2.ex_put_coord_names(exo,["X","Y","Z"])
    
    ex2ndx = 1
    for v in range( 1,len(vars) ):
        var  = vars[v]
        ncmp = varcmp[v]
        if ( ncmp == 1 ):
            oset = 0
            var  = var.reshape((var.size,1))
        else:
            oset = ncmp -1
        for d in range(ncmp):
            data = var[:,oset-d]
            ex2.ex_put_nodal_var(exo,1,ex2ndx,nnodes,data)
            ex2ndx = ex2ndx +1        
        
    ex2.ex_close(exo)

    return ofpath


#################################################################
#
def ndvel(pd,vkey,epslc):
    """
    ----
    
    pd              # PIVData object containing the velocity data.
    vkey            # Velocity variable name within pd.
    epslc           # Epoch slice.

    ----

    Non-dimensionalizes the velocity values by cell size:
        ndvel = vel/cellsz
    
    Returns ndvpd, a PIVData object containing the non-dimensionalized 
    velocities.  Only the velocities for the Epochs in epslc will actually 
    be normalized.  The normalized velocities will be stored under the 
    variable name NDV.
    """
    # Initialization.
    pd = pd[epslc]
    
    ndvpd = pivdata.PIVData(pd.cellsz,pd.origin,"NDVPD")
    
    # Get non-dimensionalization factor.
    ncells = pd[0].eshape
    cellsz = pd.cellsz
    ndf    = empty((3,ncells[0],ncells[1],ncells[2]),dtype=float)
    for i in xrange(3):
        ndf[i,...] = 1./cellsz[i]
        
    # Non-dimensionalize.
    for e in xrange(len(pd)):
        ndv = pivdata.cpivvar(ndf*pd[e][vkey],"NDV","NA")
        ndvpd.addVars(e,ndv)
        
    ndvpd.setTimes(pd.getTimes())

    return ndvpd


#################################################################
#
def psvtrace(
    pd, vkey, icomp, ntrpc, tssdiv, src=None, adjid=False, hist=0,
    rsd=0, ctcsp=True, tcfun=None, epslc=None, writer=None,
    interp=['L','L'], ndpd=True
    ):
    """
    ----

    pd               # PIVData object containing the velocity data.
    vkey             # Velocity variable name within pd.
    icomp            # Initial composition PIVVar or callback function.
    ntrpc            # Number of tracers per cell.
    tssdiv           # Number of subtime steps for each Epoch time step.
    src=None         # (nz,ny,nz) array specifying source cells.
    adjid=False      # Flag to force tracers to source ID in source cells.
    hist=0           # Flag to return tracer coordinates for each step.
    rsd=0            # Random seed.
    ctcsp=True       # Coerce tracer coordinates to single precision.
    tcfun=None       # Tracer coordinate adjustment function.
    epslc=None       # Epoch slice.
    writer=None      # File writer callback function.
    interp=['L','L'] # Interpolation method [Space,Time].
    ndpd=True        # Non-dimensionalize velocities in pd.

    ----

    Advects passive tracers through time using the velocity in pd
    with key vkey.  

    NOTE: Although the technique employed is documented here,
    users should call mpsvtrace() instead of psvtrace().

    Tracer advection is undertaken using a time-varying velocity field 
    that is contained within PIVEpochs and stored in the PIVData object 
    pd.  Tracers, having a particular identity are advected through the 
    velocity field using a fourth-order Runge-Kutta scheme at time steps 
    of size edt/tssdiv, where edt is the time between two Epochs in the 
    PIVData object.  Tracers will be advected starting at Epoch 0 and
    continuing until the last Epoch in pd, unless epslc is set to a valid
    python slice (see below).  The data returned by psvtrace() is an array
    of advected tracer coordinates at the end of advection and an
    additional array specifying the ID of each tracer.

    NOTE: Tracer coordinates are normalized by cellsz (ie, they are
    in grid units, not mm).  A coordinate of (0,0,0) is located perfectly 
    in the center of the (0,0,0) cell.   

    The velocity at the spatial location of each tracer is computed using
    the interpolation method specified by interp[0].  The value of
    interp[0] can be one of two values:
        'L' ---- Use trilinear spatial interpolation.
        'C' ---- Use tricubic spatial interpolation.
    Trilinear interpolation is cheap and often produces acceptable
    results, but errors can be large in regions of high curvature (ie,
    where spatial second derivatives of the velocity field are large).
    Tricubic interpolation is based on the technique of Lekien:2005 which
    gives a C1 continuous interpolated velocity field.  For velocity
    fields with strong spatial second derivatives, the use of tricubic
    interpolation can significantly reduce errors in advected tracer
    coordinates.  Unfortunately, tricubic interpolation is very expensive
    both computationally and memory-wise.  The computational burden alone
    can generally be expected to increase run times by ~3X over trilinear
    interpolation.

    In a somewhat similar manner to spatial interpolation, the velocity
    field at each time step can be interpolated using either linear
    or cubic spline interpolation.  The user's choice is specified via
    interp[1] which can take either of two values:
        'L' ---- Use linear temporal interpolation.
        'C' ---- Use cubic spline temporal interpolation.
    Again, linear interpolation in much cheaper in terms of computational
    cost and memory than cubic spline interpolation, while cubic spline
    interpolation provides a higher-order approximation to the data.

    psvtrace() tries to be as flexible as possible in permitting the user
    to undertake Lagrangian studies of flow fields stored on a structured 
    mesh.  Beyond the basic advection of some initial distribution of 
    tracers, psvtrace() provides additional tools for tracer advection.
    
    1) Tracer coordinates can be passed directly by the user via the
    tcfun argument (see below), or psvtrace() can compute the tracer
    coordinates based on the number of cells in the computational domain
    as well as the ntrpc parameter.  
    
    The advantage of letting psvtrace() construct the tracer coordinates 
    (and corresponding tracer ID) is the simplicity provided to the user.  
    As dicussed below, the user can still intercept, via the tcfun
    parameter, the set of auto generated coordinates prior to advection.

    The advantages of passing an array of tracer coordinates directly
    to psvtrace() lie primarily with memory utilization and overall
    flexibility of specifying exactly what tracers are of importance
    from the beginning.  When psvtrace() constructs the tracer coordinate
    array, it assumes each cell gets ntrpc as described below.  While
    the user can then prune or otherwise modify this set of tracer
    coordinates via the tcfun parameter, resources must still be allocated
    to hold the full array.  If the user is only interested in the 
    behavior of a batch of tracers initially located in a limited 
    region of the full domain, it can be much more efficient for the user
    to provide coordinates (and tracer ID's) directly.

    2) The initial composition of all cells in the flow field can
    be specified via icomp.  icomp can be:
        - a PIVVar giving the initial composition for each cell in
          the velocity mesh,
        - a callback function that the user can provide to
          interactively set the tracer composition prior to advection,
        - or an array of length totaltracers that gives the tracer
          ID for each individual tracer.  This option is only available
          if the tracer coordinate array is passed directly via the
          tcfun parameter.
    
    If icomp is set to a PIVVar, the PIVVar must be of shape (1,nz,ny,nx), 
    with nz, ny, and nx representing the number of cells in the flow domain.
    The starting tracer distribution is then constructed as follows.  Each 
    cell is initially given ntrpc tracers that are distributed throughout 
    the cell uniformly or randomly depending on the value of ntrpc.  The ID 
    of each tracer is set to the icomp value for the cell.  

    icomp can be set to a callback function that will be used to
    set the initial tracer ID.  The principle advandatage of this approach
    is that individual tracers can be assigned composition ID's with
    arbitrary spatial resolution (as opposed to all tracers within a
    single cell having the same composition).  The signature for the 
    callback function must be
        trid = fun(tcrd,pd)
    where tcrd is the initial tracer coordinate array and pd is the 
    PIVData object passed to psvtrace().  Recall that tracer coordinates
    will be expressed in dimensionless units of cells.  fun() must
    return a 1D numpy array, trid, of tracer ID's for each tracer in
    tcrd. The recommended dtype for trid is an int16 array (ie, 
    dtype=int16).
    
    NOTE: If tcfun is set to a function, then icomp must be set to a
    function as well.  See below.    
    
    If the user intends to pass the initial tracer coordinates directly
    via tcfun, then the corresponding tracer ID's can be passed directly
    to psvtrace() by way of icomp.
    
    In all three of the above cases, unless adjid is True, the ID of a 
    particular tracer is a constant for all time.
    
    3) If ntrpc is an integer, then the tracers will be distributed randomly
    as described below.  ntrpc can also be set to a three element list 
    containing the number of tracers to be used in the z, y, and x 
    directions of each cell.  In this case, tracer location is not 
    randomized.  Instead, tracers will be distributed uniformly throughout 
    the cell with ntrpc tracers along each cell axis.
    
    NOTE: Unless source cells are specified via src, ntrpc is ignored if 
    tcfun is set to an array.

    4) psvtrace() permits the user to specify particle sourcing cells
    by way of src.  The shape of src is (nz,ny,nx) or (1,nz,ny,nx), but 
    src does not have to be a PIVVar.  Any src cell with a value greater 
    than -1 will generate new tracer particles at each time step during 
    advection only when the number of tracer particles within the 
    corresponding cell drops below ntrpc.  These new tracer particles
    will be appended to the end of the tcrd array.  The identity of the new 
    source particle will be equal to the value of src for that cell.  If no 
    src variable is specified, then no new source particles will be 
    generated.  Because of the memory intensive nature of tracer advection, 
    it is recommended that src be an int16 array (ie, dtype=int16).  
    
    5) psvtrace() allows the user to force the ID of all particles 
    within a source cell to that of the source cell via adjid.  When adjid
    is True, all source cells will be interrogated at each time step, and
    the particles within will take the identity of the source cell as
    specified in src.

    NOTE: If adjid is True, then src must be specified.

    6) psvtrace() and the underlying advection facilities use double
    precision floating point arithmetic to preserve as much precision as
    possible during tracer advection.  However, indefinitely preserving 
    tracer coordinates with such precision is probably not needed, and
    significant memory/hard disk savings can be realized by reducing
    tracer coordinate precision after advection.  If ctcsp is True, then
    tracer coordinates will be stored as single precision floats after
    advection.  To be clear, tracer advection will still be done using 
    double precision arithmetic regardless of ctcsp.  The coordinates 
    will simply be converted to single precision after advection if ctcsp
    is True.  

    7) hist permits the user to retrieve a list of all tracer coordinates
    from particular time steps, depending on the value of hist.  hist
    can be set to 3 values:
        0 ---- Only the initial and final tcrd and trid variables will be 
               returned.
        1 ---- tcrd and trid will be returned for each Epoch in pd.
               Variables will be output at the start of each
               Epoch.
        2 ---- tcrd and trid will be returned for each time step
               (including tssdiv).  An additional variable, vtmsk, will
               also be returned.  Variables will be output immediately 
               after the call to RK4 but before any source particles are 
               added.
        3 ---- This mode is a combination of modes 1 and 2.  tcrd and 
               trid along with vtmsk will be returned for each Epoch in 
               pd.  These three variables will be output at the start of 
               each Epoch.  NOTE: Source cells cannot be used with 
               hist = 3.
    Note that tcrd is the array of tracer coordinates and trid is the 
    array of tracer ID's.
     
    The vtmsk array returned when hist = 2 or 3 contains a boolean flag
    representing valid tracers in the tcrd array in moving from the 
    current time step to the next.  vtmsk can be used to track individual
    particles through time.  Consider a tcrd array at timestep tsi that
    is advected to the next timestep tsi+1.  At tsi+1, some particles in
    tcrd may have advected beyond the perimeter.  These particles will
    be flagged as invalid and removed from tcrd(tsi+1).  The 
    correspondence between old and new tcrd arrays then becomes
        tcrd(tsi)[vtmsk,:] --> tcrd(tsi+1) 
    If source cells are used, then tcrd(tsi+1) will be augmented by
    appending new tracers to the tcrd(tsi+1) array.
    
    NOTE: Be very careful with hist as it will easily run the process
    out of memory unless writer parameter is set to a valid function!

    NOTE: If the writer parameter is set to a valid function, then
    the above variables will not be returned.  Instead, the writer
    function will be called at the appropriate times.  See below.

    8) In many cases, the tracer data produced by psvtrace() are not
    needed all at once.  Consequently, storing these arrays to disk
    for later processing by the user can free up much needed memory.
    The writer parameter permits the user to specify a callback function
    to write tracer data instead of returning it.  psvtrace() still
    behaves as described elsewhere and still generates the same data.
    It simply writes it to file at the earliest possible opportunity.
    The signature for the writer callback must be
        writer(vars,epoch,ts)
    where epoch is the current Epoch tracers are being advected from
    and ts is the current sub timestep (as specified via tssdiv).
    psvtrace() will ignore any value returned by the writer function.
    The writer function will be called at the same points in the
    tracer code that variables would be saved (see notes on the hist
    parameter above).
    
    9) A function can be passed via tcfun that will be called by
    psvtrace() to adjust tracer coordinates.  tcfun must have the
    signature
        tcrd = tcfun(tcrd,ntrpc,pd)
    where tcrd is the initial tracer coordinate array (computed using
    rsd), ntrpc is the same as for psvtrace(), and pd is the PIVData 
    object passed to psvtrace().  tcfun can modify tcrd in any way.  For
    example, tcfun could prune the initial tracer coordinates so that 
    only those tracers within a certain region of interest are advected.
    The returned tcrd must be a numpy array with the shape 
    tcrd.shape = (n,3), where n is the number of tracers.
    
    NOTE: If tcfun is set to a function, then icomp must be set to a
    function as well.
    
    NOTE: tcfun will only be called prior to starting advection.
    
    As discussed above, however, tcfun can also be directly set to the 
    array of initial tracers coordinates that psvtrace() should advect.
    In this case, tracer ID's can be set by any of the three methods
    discussed above for the icomp parameter. 

    10) psvtrace is constructed such that multiple calls using the same
    arguments produces identical results even though a random number
    generator is used for tracer coordinate initialization.  Identical
    results are enforced by seeding the random number generator with
    integers, starting at rsd and incrementing by one each time 
    the RNG is called.
    
    11) If epslc is set to a valid python slice, then psvtrace() will advect
    tracers using only the Epochs specified in the slice.  Note that if the
    slice step is negative, then passive tracers will be advected backward
    in time (eg, slice(10,5,-1)).  Set epslc=None to advect tracers from 
    Epoch 0 to the last Epoch in pd.
    
    Note: To use Epoch 0 when advecting in reverse time, set the slice
    stop index to None.  For example,  
        slice(10,None,-1)
    will reverse advect tracers starting at Epoch 10 and finishing at
    Epoch 0.
    
    Note: psvtrace() employs python slices directly so epslc has the same
    effect on psvtrace() as a slice on any array.  Setting
        epslc = slice(4,8,1)
    will advect tracers starting at Epoch 4 and stopping on Epoch 7
    (ie, 4->5, 5->6, and 6->7).
    
    12) Non-dimensionalizing the velocity field can be a time consuming
    operation for datasets with a large number of Epochs.  The user may
    wish to pass a previously non-dimensionalized PIVData object directly 
    into psvtrace() to eliminate this overhead.  To do so, simply set the
    ndpd flag to False, and psvtrace() will assume the velocity field of
    pd is already non-dimensionalized. 

    13) psvtrace() prints diagnostic information regarding tracer count at
    each sub iteration.  The output varies depending on whether source
    cells are specified.  The possible output fields are:
        IT ----- Sub iteration number.  Runs from 0 .. tssdiv.
        TNTR --- Total number of tracers that will be advected during the
                 the sub iteration.
        VNTR --- Number of 'valid' tracers following the advection.
                 Valid tracers are those that lie within the perimeter
                 of the data grid.  If source cells are not used, VNTR
                 is the total number of tracers at the end of the advection
                 step.
        BYP ---- Number of particles that were found to be beyond the
                 the perimeter.  BYP = TNTR - VNTR
        NNST --- Number of new source tracers added to source cells such
                 that the number of particles within the cells is equal to
                 ntrpc.
        XTNTR -- Total number of tracers at the end of the advection step
                 when source cells are specified.  XTNTR = VNTR + NNST 


    In summary, psvtrace() returns a list of the following form
        [ [tcrd(ts0),trid(ts0),vtmsk(ts0)],
          [tcrd(ts1),trid(ts1),vtmsk(ts1)], ... ] 
    where tcrd is the tracer coordinate array, and has the shape
    (totaltracers,3).  Each tracer's coordinates are specified in the 
    order (z,y,x).  trid is the corresponding tracer ID array, and
    vtmsk is the valid tracer boolean mask (which will only be output
    if hist = 2).       
    """
    print "STARTING: psvtrace"

    # Initialization.
    if ( ( compat.checkNone(src) ) and ( adjid ) ):
        raise ValueError("adjid cannot be True if no source cells specified.")

    cellsz = pd.cellsz
    origin = pd.origin

    ncells  = array( pd[0][vkey].shape[1:4] )
    tncells = ncells[0]*ncells[1]*ncells[2]

    rbndx = zeros([3,2],dtype=int)
    for i in range(3):
        rbndx[i,:] = [0,ncells[i]]

    netstps = len(pd)   

    # Perform some checks.
    havetcrd = False    
    if ( callable(tcfun) ):
        if ( not callable(icomp) ):
            raise ValueError("If tcfun is a function, icomp must be a function.")
    elif ( compat.checkNone(tcfun) ):
        if ( not callable(icomp) and not isinstance(icomp,pivdata.PIVVar) ):
            raise ValueError("icomp set to an array but tcfun is not.")
    else:
        tcrd     = array(tcfun)
        havetcrd = True

    havetrid = False
    if ( not callable(icomp) and not isinstance(icomp,pivdata.PIVVar) ):
        trid     = array(icomp)
        havetrid = True
        
    if ( not compat.checkNone(src) and hist == 3 ):
        raise ValueError("Source cells cannot be used with hist = 3.")

    # Set tracer coordinate dtype.
    if ( ctcsp ):
        tcdtype = float32
    else:
        tcdtype = float

    # Get tracer coordinates.
    ntrpc  = array(ntrpc)
    sntrpc = ntrpc.prod()
    if ( not havetcrd ):
        if ( ntrpc.size > 1 ):
            print " | Tracers per cell: %i, %i, %i" % tuple(ntrpc)
        else:
            print " | Tracers per cell: %i" % ntrpc
    
        if ( ntrpc.size > 1 ):
            tcrd = gettcrds(ncells,ntrpc,None)
        else:
            tcrd = gettcrds(ncells,ntrpc,rsd)        
        rsd  = rsd +2
    
        if ( callable(tcfun) ):
            print " | Calling tcfun()."
            tcrd = tcfun(tcrd,ntrpc,pd)

    tntr = tcrd.shape[0]

    # Set the tracer id.
    if ( callable(icomp) ):
        print " | Calling icomp()."
        trid = icomp(tcrd,pd)
    elif ( not havetrid ):
        ctcrd = (tcrd +0.5).astype(int)
        trid  = icomp[0,ctcrd[:,0],ctcrd[:,1],ctcrd[:,2]]
        del ctcrd
        
    # Store the initial data.  Take a copy since some functions below
    # modify tcrd in place.
    thlst = []
    thlst.append([tcrd.astype(tcdtype).copy(),trid.copy()])

    # Setup ntic.
    ntic = empty((ncells[0],ncells[1],ncells[2]),dtype=int)
    ntic[:,...] = sntrpc

    # Set up source cells.
    if ( not compat.checkNone(src) ):
        msk = src > -1
        msk = msk.reshape(tncells)
        nsc = msk.sum()

        ndx    = indices(ncells).reshape(3,tncells)
        ndx    = ndx.transpose()
        scndx  = ndx.compress(msk,axis=0)
        sccomp = src.reshape(tncells)
        sccomp = sccomp.compress(msk)
        
        del msk, ndx

    # Process epslc.  A python slice object can have None for start, 
    # stop, and step members.  The easiest way to remove the None
    # values is to build a simple arange array and slice the array.
    if ( compat.checkNone(epslc) ):
        epslc = slice(0,len(pd),1)

    proc   = arange(len(pd))
    proc   = proc[epslc]
    emn    = proc.min()
    emx    = proc.max()
    etstep = epslc.step

    if ( compat.checkNone(etstep) ):
        etstep = 1
    elif ( abs(etstep) > 1 ):
        raise ValueError("|epslc.step| > 1")
    
    if ( etstep > 0 ):
        etstart = emn
        etstop  = emx
        stepdir = 1

    else:
        etstart = emx
        etstop  = emn
        stepdir = -1

    # Non-dimensionalize the velocities by the cell size.
    if ( ndpd ):
        ndvpd  = ndvel(pd,vkey,slice(0,len(pd)))
        ndvkey = 'NDV'
    else:
        ndvpd  = pd
        ndvkey = vkey

    # Set up fargs parameter list based on interp.
    fargs = initfargs(pd,ndvpd,ndvkey,interp)

    # Advection loop.
    print " | Advecting tracers ..."
    
    hcnt = 0
    if ( ( not compat.checkNone(writer) ) and hist < 2 ):
        writer(thlst[0],etstart,0)
        thlst.pop(0)        
    for et in xrange(etstart,etstop,etstep):
        # Setup fargs for rk4().
        if ( stepdir > 0 ):
            print " | Epoch %i -> %i" % (et,et+1)            
        else:
            print " | Epoch %i -> %i" % (et,et-1)

        fargs[0] = ndvpd[et][ndvkey]
        fargs[1] = ndvpd[et+stepdir][ndvkey]
        fargs[2] = pd[et].time
        
        edt      = pd[et+stepdir].time -pd[et].time
        idt      = edt/tssdiv            
        fargs[4] = edt

        # Store tcrd from previous Epoch and write results if necessary.
        if ( hcnt > 0 ):
            if ( hist == 1 ):                
                thlst.append([tcrd.astype(tcdtype).copy(),trid.copy()])
                if ( not compat.checkNone(writer) ):
                    writer(thlst[0],et-1,0)
                    thlst.pop(0)        

            elif ( hist == 3 ):
                thlst.append([tcrd.astype(tcdtype).copy(),trid.copy()])

                evtmsk[etrndx] = True
                if ( not compat.checkNone(writer) ):
                    thlst[0].append(evtmsk)
                    writer(thlst[0],et-1,0)
                    thlst.pop(0)        
                else:
                    thlst[hcnt -1].append(evtmsk)
                    
        # Initialize evtmsk and etrndx for hist=3.  Need to initialize
        # both evtmsk and etrndx here since the tntr will be reduced
        # during the iterations.
        if ( hist == 3 ):
            evtmsk = zeros(tntr,dtype=bool)
            etrndx = arange(tntr)

        for it in range(tssdiv):  
            fargs[3] = it*idt

            tcrd     = floutil.rk4(tcrd,idt,velfun,fargs)

            # Throw out particles that have advected beyond the perimeter.
            [cta,ctandx,vtndx] = flotracec.ctcorr(tcrd,ncells,rbndx)

            vntr         = vtndx.size
            trid[0:vntr] = trid[vtndx]

            if ( hist == 2 ):
                thlst.append([tcrd[0:vntr].astype(tcdtype).copy(),
                              trid[0:vntr].copy()])
                
                vtmsk        = zeros(tntr,dtype=bool)
                vtmsk[vtndx] = True

                if ( not compat.checkNone(writer) ):
                    thlst[0].append(vtmsk)
                    writer(thlst[0],et,it)
                    thlst.pop(0)
                else:
                    thlst[hcnt].append(vtmsk)

                hcnt  = hcnt +1
            elif ( hist == 3 ):
                etrndx = etrndx[vtndx]

            if ( adjid ):
                flotracec.adjid(trid,cta,ctandx,scndx,sccomp)

            # Replinish source cells.
            if ( not compat.checkNone(src) ):
                [stndx,stid,nnst] = flotracec.chksrc(ctandx,scndx,sccomp,sntrpc)
                
                xtntr = vntr +nnst
                print \
                    " |-| IT %3i, TNTR %i, VNTR %i, BYP %i, NNST %i, XTNTR %i"\
                    % (it,tntr,vntr,tntr-vntr,nnst,xtntr)

                if ( xtntr != tntr ): 
                    ntcrd = zeros((xtntr,3),dtype=float)
                    ntcrd[0:vntr,:] = tcrd[0:vntr,:] 

                    tcrd = ntcrd               
                    trid = resize(trid,xtntr)

                for i in range(3):
                    rsd = rsd +1
                    random.seed(rsd)

                    tcrd[vntr:xtntr,i] = stndx[0:nnst,i] +random.rand(nnst) -0.5

                trid[vntr:xtntr] = stid[0:nnst]
                tntr = trid.size
                    
            else:
                print " |-| IT %3i, TNTR %i, VNTR %i, BYP %i" %\
                    (it,tntr,vntr,tntr-vntr)

                tcrd = resize(tcrd,(vntr,3))
                trid = resize(trid,vntr)
                
                tntr = vntr

        if ( hist != 2 ):
            hcnt = hcnt +1
            
    random.seed(None)
    print " | EXITING: psvtrace"
    if ( hist == 2 ):
        vtmsk = ones(vntr,dtype=bool)

        if ( not compat.checkNone(writer) ):
            thlst[0].append(vtmsk)
        else:
            thlst[hcnt].append(vtmsk)
    elif ( hist == 3 ):
        thlst.append([tcrd.astype(tcdtype),trid])
        vtmsk = ones(vntr,dtype=bool)
        thlst[-1].append(vtmsk)
        
        evtmsk[etrndx] = True
        if ( not compat.checkNone(writer) ):
            thlst[0].append(evtmsk)
            writer(thlst[0],etstop-1,0)
            thlst.pop(0)        
        else:
            thlst[hcnt -1].append(evtmsk)        
    else:
        thlst.append([tcrd.astype(tcdtype),trid])

    if ( not compat.checkNone(writer) ):
        writer(thlst[0],etstop,0)
    else:
        return thlst


#################################################################
#
def rpsvtrace(
    pd, vkey, icomp, tssdiv, csdiv=(1,1,1), src=None, rbndx=None,
    cinterp=False, cisep=None, interp=['L','L'], ndpd=True
    ):
    """
    ----

    pd               # PIVData object containing the velocity data.
    vkey             # Velocity variable name within pd.
    icomp            # (1,nz,ny,nx) PIVVar containing initial composition.
    tssdiv           # Number of subtime steps for each Epoch time step.
    csdiv=(1,1,1)    # Composition subdivision factor (z,y,x).
    src=None         # (nz,ny,nx) array specifying source cells.
    rbndx=None       # Region bounding indices.
    cinterp=False    # Composition interpolation flag.
    cisep=None       # Continuous integration starting Epoch.
    interp=['L','L'] # Interpolation method [Space,Time].
    ndpd=True        # Non-dimensionalize velocities in pd.

    ----

    Advects passive tracers backward through time using the velocity in pd
    with key vkey.  

    rpsvtrace() is identical to psvtrace() in many respects, except that
    rpsvtrace() advects the tracers backward through time.  This reverse
    advection occurs in one of two ways based on the value of cisep.  
    
    | cisep = None | At each Epoch, eg. E(i+1), cells are assigned a single
    tracer particle with an unknown ID.  This particle is then backward 
    advected tssdiv time steps to Epoch E(i), where it will be occupying 
    a new cell.  The tracer ID, and hence cell composition, at time E(i+1)
    is then determined by the composition of the cell the tracer occupies
    at E(i) according to the cinterp flag.  If the tracer moves beyond the
    perimeter, it is assigned the composition of the last source cell it 
    crosses (if any) or -1.

    | cisep = Epoch Index | Here, reverse advection proceeds similarly
    to the cisep=None case except that the compositional field is not
    recomputed at each Epoch.  Instead, tracers are reverse advected
    completely from Epoch cisep to Epoch 0, and the compositional field
    for Epoch cisep is then determined.  Note that the compositional 
    field for the intermediate Epochs is not computed.  As a result, the
    returned PIVData object will contain composition fields for only two 
    Epochs: Epoch 0 and Epoch cisep.  The principal advantages of this
    approach over the cisep=None case are the reduction of aliasing and
    spurious diffusion errors.  As explained below, interpolation serves 
    to reduce aliasing that arises when advecting the compositional field,
    and is highly recommended when cisep=None.  The diffusion error
    introduced by cinterp=True for the cisep=None method could be
    objectionable in some cases.  Setting cisep to a particular Epoch
    will greatly reduce one type of aliasing error.  Use of interpolation
    is still recommended, however when cisep is set to an Epoch index,
    interpolation is done only once during the whole scheme.  So diffusion
    errors will be minimal.  One final note: aliasing will likely still
    be present when setting cisep to a value other than None.  In this
    case, however, the aliasing is generally limited to jagged edges
    around the compositional boundaries which can be removed by increasing
    csdiv. 

    If cinterp is True, then the tracer ID will be computed from the
    neighboring cells using trilinear interpolation.  If cinterp is False,
    the tracer will be assigned the ID of the cell it lands in during
    reverse advection.  Interpolation acts as a smoothing filter and 
    eliminates issues with aliasing that can arise without interpolation.
    The disadvantage to interpolation is that compositional boundaries will
    be blurred.  Use of interpolation is recommended.

    The method is motivated by two primary factors.  First, forward
    advection of tracers is sensitive to flow field noise which can result
    in an uneven distribution of tracers in some cells.  Overcoming the
    first concern leads to the second.  Namely, a large number of particles
    must be used for forward advection to ensure that all cells end up with
    at least one particle.  For more information on reverse techniques, 
    see Sutton:1994.

    Bacward advection eliminates both of the issues with forward advection.  
    At every Epoch, a cell is guaranteed to have 1 particle, and at any
    given Epoch or smaller timestep, the number of tracers being tracked
    is equal to the number of cells.

    rpsvtrace() differs from psvtrace() in a few other areas as well. 
    Perhaps most importantly, rpsvtrace() does not need to compute
    composition, per se.  Composition is assiged as described earlier.

    Although the velocity field is specified on a particular grid,
    Lagrangian particles are not confined to coordinates that are 
    discretized like the velocity grid.  Clearly the level of compositional
    detail that can be ultimately resolved is a direct function of velocity
    resolution.  However within a given velocity field of arbitrary 
    resolution, Lagrangian particles are free to assume any valid, 
    continuous coordinates.  As a result, rpsvtrace() provides a mechanism
    to increase the composition resolution via csdiv.  If csdiv > 1 for a
    particular axis, then the computed composition field with be broken
    into csdiv subcells along that axis. 

    Also in a similar fashion to psvtrace(), rpsvtrace() provides the
    user with the ability to specify sourcing cells by way of src.  
    In rpsvtrace() any particle that lands in a source cell at a timestep
    is assigned the composition of that cell.  If while advecting backward
    in time the tracer crosses multiple source cells, it will be assigned the 
    composition of the last source cell it crosses.  If while advecting the
    tracer crosses multiple source cells and lands in a non-source cell, the
    particle will take the ID of the non-source cell but will maintain the
    source cell ID if the tracer advects beyond the computational perimeter
    in a subsequent timestep.  Source cells are identified by setting src
    to a value greater than -1 for the desired cell.  Generally speaking,
    the value of src (for a particular source cell) should match that of
    icomp.

    Tracer advection is undertaken using the same underlying RK4 methods
    as psvtrace().

    The velocity at the spatial location of each tracer is computed using
    the interpolation method specified by interp[0].  The value of
    interp[0] can be one of two values:
        'L' ---- Use trilinear spatial interpolation.
        'C' ---- Use tricubic spatial interpolation.
    Trilinear interpolation is cheap and often produces acceptable
    results, but errors can be large in regions of high curvature (ie,
    where spatial second derivatives of the velocity field are large).
    Tricubic interpolation is based on the technique of Lekien:2005 which
    gives a C1 continuous interpolated velocity field.  For velocity
    fields with strong spatial second derivatives, the use of tricubic
    interpolation can significantly reduce errors in advected tracer
    coordinates.  Unfortunately, tricubic interpolation is very expensive
    both computationally and memory-wise.  The computational burden alone
    can generally be expected to increase run times by ~3X over trilinear
    interpolation.

    In a somewhat similar manner to spatial interpolation, the velocity
    field at each time step can be interpolated using either linear
    or cubic spline interpolation.  The user's choice is specified via
    interp[1] which can take either of two values:
        'L' ---- Use linear temporal interpolation.
        'C' ---- Use cubic spline temporal interpolation.
    Again, linear interpolation in much cheaper in terms of computational
    cost and memory than cubic spline interpolation, while cubic spline
    interpolation provides a higher-order approximation to the data.

    The initial composition of all cells in the flow field is specified
    via icomp.  icomp must be a PIVVar of shape (1,nz,ny,nx), with nz, ny,
    and nx representing the number of cells in the velocity variable
    multiplied by csdiv.  A composition of any value other than -1 can be 
    specified by the user.
    
    The shape of src is (nz,ny,nx) or (1,nz,ny,nx), where nz, ny, and nx
    are the number of cells in the icomp PIVVar.  Note, however, that src 
    does not have to be a PIVVar.  

    The user can limit tracing to a specified region of the flow field
    via the 3x2 array rbndx.  rbndx contains the indices of the region
    in python slice notation
        rbndx[0,0] -- zstart
        rbndx[0,1] -- zstop +1
        rbndx[1,0] -- ystart
        rbndx[1,1] -- ystop +1
        rbndx[2,0] -- xstart
        rbndx[2,1] -- xstop +1
        
    NOTE: rpsvtrace() can consume a large amount of memory depending on 
    the values of csdiv, and can easily run out of address space on 32
    bit machines.  It is highly recommended that the user pass icomp and 
    src with a numpy dtype of int16.  Whatever dtype is used with icomp will
    be used for new composition arrays created by rpsvtrace().

    NOTE: If the user passes icomp and src as int16 arrays as recommended 
    above and also activates interpolation, then the discrete composition
    values specified in icomp and src must have sufficient room between
    them to permit succesfull interpolation.  As an example, consider a
    case where three composition layers are initially specified.  One could
    set these to 0,1,2 without interpolation.  But with interpolation, they
    should be set to 0,16383,32767.  Of course, if floating point arrays
    for icomp and src are used, then there is no restrictions on composition
    values regardless of the cinterp flag.

    rpsvtrace() prints diagnostic information regarding tracer count prior
    to advection:
        TNTR --- Total number of tracers for each Epoch that will be
                 advected.
        ZTNTR -- To minimize the size of arrays that must be allocated,
                 tracers in each z-plane are advected separately.  ZTNTR
                 represents the number of tracers in a given z-plane.
 
    rpsvtrace() returns a new PIVData object containing the computed 
    composition.
       
    """
    print "STARTING: rpsvtrace"

    # Initialization.
    csdiv   = array(csdiv)
    vcellsz = pd.cellsz
    vorigin = pd.origin

    fnvcells = array( pd[0][vkey].shape[1:4] )
    if ( compat.checkNone(rbndx) ):
        subrgn  = False
        nvcells = fnvcells
    
        rbndx = zeros((3,2),dtype=int)
        for i in range(3):
            rbndx[i,:] = [0,nvcells[i]]
    else:
        subrgn  = True
        rbndx   = array(rbndx)
        nvcells = rbndx[:,1] -rbndx[:,0]

    tnvcells = nvcells[0]*nvcells[1]*nvcells[2]

    ntcells = csdiv*nvcells

    # To minimize the size of arrays that must be allocated, tracers
    # on each z-plane will be advected separately.
    tntr  = ntcells[0]*ntcells[1]*ntcells[2]
    ztntr = ntcells[1]*ntcells[2]

    if ( not isinstance(icomp,pivdata.PIVVar) ):
        raise TypeError("icomp must be a PIVVar.")

    compname  = icomp.name
    compunits = icomp.units
    compvtype = icomp.vtype
    cdtype    = icomp.dtype

    floatcomp = (cdtype.name).upper().find("FLOAT") >= 0

    # Determine how to process the epochs.
    if ( compat.checkNone(cisep) ):
        etstart = 1
        etstop  = len(pd)
    else:
        etstart = cisep
        etstop  = cisep +1

    # Non-dimensionalize the velocities by the cell size.
    if ( ndpd ):
        ndvpd  = ndvel(pd,vkey,slice(0,etstop))
        ndvkey = 'NDV'
    else:
        ndvpd  = pd
        ndvkey = vkey

    # Set up fargs parameter list based on interp.
    fargs = initfargs(pd,ndvpd,ndvkey,interp)

    # Get tracer id.
    trid = icomp.reshape(ntcells)  

    # Setup tpd[0].
    tcellsz = vcellsz/csdiv
    orgofst = rbndx[:,0]*vcellsz
    torigin = vorigin +orgofst -(csdiv -1.)*tcellsz/2.  
    tpd     = pivdata.PIVData(tcellsz,torigin,pd.desc)
    tpd.addVars(0, pivdata.cpivvar(trid,compname,compunits,compvtype) )

    # Set up source cell arrays.
    if ( not compat.checkNone(src) ):
        fntcells = csdiv*fnvcells
        src      = src.reshape(ntcells)
        xrbndx   = rbndx.copy()
        for i in range(3):
            xrbndx[i,:] = xrbndx[i,:]*csdiv[i]

        fsrc = zeros(fntcells,dtype=cdtype)
        fsrc[:,...] = -1.
        
        fsrc[xrbndx[0,0]:xrbndx[0,1],
             xrbndx[1,0]:xrbndx[1,1],
             xrbndx[2,0]:xrbndx[2,1]] = src

        msk = src > -1. 
        msk = msk.reshape(tntr)
        
        scndx  = arange(tntr,dtype=int)
        scndx  = scndx.compress(msk)
        sccomp = src.reshape(tntr)
        sccomp = sccomp.compress(msk)

    print " | TNTR %i, ZTNTR %i" % (tntr,ztntr)

    # Advection loop.
    print " | Advecting tracers ..."
    for et in range(etstart, etstop):
        print " | Epoch %i" % et

        comp    = empty(tntr,dtype=cdtype)
        comp[:] = -1.
        for tz in range(ntcells[0]):
            # Get initial tracer coordinates.
            tcrd = gettcrds([1,nvcells[1],nvcells[2]],
                            [1,csdiv[1],csdiv[2]])
                            
            tcrd[:,0] = tcrd[:,0] +float(tz)/csdiv[0] \
                                  -(csdiv[0] -1.)/(2.*csdiv[0])

            if ( subrgn ):
                for i in range(3):
                    tcrd[:,i] = tcrd[:,i] +rbndx[i,0]
        
            cvtndx  = arange(ztntr,dtype=int) +tz*ztntr

            # Advect.
            etsstart = et
            if ( compat.checkNone(cisep) ):
                etsstop = et -1
            else:
                etsstop = 0
            
            for ets in range(etsstart,etsstop,-1):
                edt = pd[ets-1].time -pd[ets].time
                idt = edt/tssdiv
        
                fargs[0] = ndvpd[ets][ndvkey]
                fargs[1] = ndvpd[ets -1][ndvkey]
                fargs[2] = pd[ets].time
                fargs[4] = edt

                for it in range(tssdiv):
                    fargs[3] = it*idt
                    tcrd     = floutil.rk4(tcrd,idt,velfun,fargs)
    
                    # Check for particles that have advected beyond the perimeter.
                    # Here, tcrd represents coordinates into the velocity arrays.
                    [cta,ctandx,vtndx] = flotracec.ctcorr(tcrd,fnvcells,rbndx)
    
                    vntr   = vtndx.size
                    cvtndx = cvtndx[vtndx]
                    tcrd   = tcrd[0:vntr]
    
                    # Set composition for particles that cross a source cell.  We
                    # need to do this here so that if a particle advects beyond the
                    # perimeter at a later time step, it gets assigned the ID of
                    # source cell it last touches (in reverse time).
                    if ( not compat.checkNone(src) ):
                        stcrd = tcrd +0.5
                        for i in range(3):
                            stcrd[:,i] = stcrd[:,i]*csdiv[i]
                        stcrd = floor(stcrd).astype(int)
    
                        comp[cvtndx] = fsrc[stcrd[:,0],stcrd[:,1],stcrd[:,2]]
    
            # Compute composition for valid tracers.  We need to convert tcrd
            # so that it represents coordinates into the expanded composition
            # array.
            if ( subrgn ):
                for i in range(3):
                    tcrd[:,i] = (tcrd[:,i] -rbndx[i,0])

            if ( cinterp ):
                for i in range(3):
                    tcrd[:,i] = (tcrd[:,i] +0.5)*csdiv[i] -0.5

                itrid = svinterp(trid,tcrd)
                if ( floatcomp ):
                    comp[cvtndx] = itrid.astype(cdtype) 
                else:
                    comp[cvtndx] = itrid.round().astype(cdtype) 
            else:
                for i in range(3):
                    tcrd[:,i] = (tcrd[:,i] +0.5)*csdiv[i]

                tcrd = floor(tcrd).astype(int)
                comp[cvtndx] = trid[tcrd[:,0],tcrd[:,1],tcrd[:,2]]

        if ( not compat.checkNone(src) ):
            comp[scndx] = sccomp

        comp = comp.reshape(ntcells)

        if ( compat.checkNone(cisep) ):
            tpd.addVars(et, pivdata.cpivvar(comp,compname,compunits,compvtype) )
            trid = comp
        else:
            tpd.addVars(1, pivdata.cpivvar(comp,compname,compunits,compvtype) )        

    etvals = pd.getTimes()
    if ( compat.checkNone(cisep) ):
        tpd.setTimes( etvals )
    else:
        tpd.setTimes( [etvals[0],etvals[cisep]] )
    print " | EXITING: rpsvtrace"
    return tpd


#################################################################
#
def pthtrace(
    pd, vkey, itcrd, tssdiv, interp=['L','L'], ndpd=True
    ):
    """
    ----

    pd               # PIVData object containing the velocity data.
    vkey             # Velocity variable name within pd.
    itcrd            # (ntrcrs,3) array containing initial tracer coordinates.
    tssdiv           # Number of subtime steps for each Epoch time step.
    interp=['L','L'] # Interpolation method [Space,Time].
    ndpd=True        # Non-dimensionalize velocities in pd.
    
    ----

    Advects passive tracers through time using the velocity in pd
    with key vkey.  

    pthtrace() functions in a very similar manner to psvtrace(), except
    that pthtrace() has no concept of composition.  It merely advects
    passive tracers and tracks their coordinates.  After tssdiv time steps, 
    the tracer coordinates are stored. A tracer will be completely removed 
    from the history if the tracer moves beyond the perimeter of the dataset
    at any timestep.

    The velocity at the spatial location of each tracer is computed using
    the interpolation method specified by interp[0].  The value of
    interp[0] can be one of two values:
        'L' ---- Use trilinear spatial interpolation.
        'C' ---- Use tricubic spatial interpolation.
    Trilinear interpolation is cheap and often produces acceptable
    results, but errors can be large in regions of high curvature (ie,
    where spatial second derivatives of the velocity field are large).
    Tricubic interpolation is based on the technique of Lekien:2005 which
    gives a C1 continuous interpolated velocity field.  For velocity
    fields with strong spatial second derivatives, the use of tricubic
    interpolation can significantly reduce errors in advected tracer
    coordinates.  Unfortunately, tricubic interpolation is very expensive
    both computationally and memory-wise.  The computational burden alone
    can generally be expected to increase run times by ~3X over trilinear
    interpolation.

    In a somewhat similar manner to spatial interpolation, the velocity
    field at each time step can be interpolated using either linear
    or cubic spline interpolation.  The user's choice is specified via
    interp[1] which can take either of two values:
        'L' ---- Use linear temporal interpolation.
        'C' ---- Use cubic spline temporal interpolation.
    Again, linear interpolation in much cheaper in terms of computational
    cost and memory than cubic spline interpolation, while cubic spline
    interpolation provides a higher-order approximation to the data.

    itcrd is the initial tracer coordinate array normalized by cellsz 
    to cell coordinates (ie, itcrd does not have units of mm).  itcrd
    has the shape (totaltracers,3).  Each tracer's coordinates are 
    specified in the order (z,y,x).  

    NOTE:  Initial tracer coordinates must be normalized by cellsz.  Integer
    coordinates lie at cell centers.

    NOTE:  The tracer coordinates returned by pthtrace() will be scaled
    by cellsz (ie, will have units of mm).

    psvtrace() prints diagnostic information regarding tracer count at
    each sub iteration.  The output varies depending on whether source
    cells are specified.  The possible output fields are:
        IT ----- Sub iteration number.  Runs from 0 .. tssdiv.
        TNTR --- Total number of tracers that will be advected during the
                 the sub iteration.
        VNTR --- Number of 'valid' tracers following the advection.
                 Valid tracers are those that lie within the perimeter
                 of the data grid.  
        BYP ---- Number of particles that were found to be beyond the
                 the perimeter.  BYP = TNTR - VNTR

    psvtrace() returns thlst, a list structure containing tracer coordinates
    for each time step.  thlst is structured as follows:
        thlst[i] ----> Provides a sublist of coordinates at each timestep
                       within a given Epoch.  For all i > 0, 
                       len(thlst[i]) = tssdiv.  For i = 0, len(thlst[0]) = 1.
        thlst[i][j] -> Provides the tracer coordinates for Epoch i and 
                       timestep j.
    The coordinates across the entire thlst structure are ordered the 
    same, so thlst[0][0][0,:] and thlst[1][2][0,:] specify the (z,y,x) 
    coordinates for tracer 0 at (Epoch 0,Timestep 0) and (Epoch 1,Timestep 2), 
    respectively.
       
    """
    print "STARTING: pthtrace"

    # Initialization.
    cellsz  = pd.cellsz
    origin  = pd.origin
    ncells  = array( pd[0][vkey].shape[1:4] )
    tntr    = itcrd.shape[0]
    netstps = len(pd)
    thlst   = [[itcrd.copy()*cellsz +origin]]     # Take a copy.
    tcrd    = itcrd

    rbndx = zeros([3,2],dtype=int)
    for i in range(3):
        rbndx[i,:] = [0,ncells[i]]

    print " | Tracers: %i" % tntr

    # Non-dimensionalize the velocities by the cell size.
    if ( ndpd ):
        ndvpd  = ndvel(pd,vkey,slice(0,netstps))
        ndvkey = 'NDV'
    else:
        ndvpd  = pd
        ndvkey = vkey
    
    # Set up fargs parameter list based on interp.
    fargs = initfargs(pd,ndvpd,ndvkey,interp)

    # Advection loop.
    print " | Advecting tracers ..."
    for et in range(1, netstps):
        print " | Epoch %i" % et

        edt = pd[et].time -pd[et-1].time
        idt = edt/tssdiv

        fargs[0] = ndvpd[et -1][ndvkey]
        fargs[1] = ndvpd[et][ndvkey]
        fargs[2] = pd[et-1].time
        fargs[4] = edt
        
        ithlst = []
        for it in range(tssdiv):  
            fargs[3] = it*idt
            tcrd     = floutil.rk4(tcrd,idt,velfun,fargs)

            # Throw out particles that have advected beyond the perimeter.
            [cta,ctandx,vtndx] = flotracec.ctcorr(tcrd,ncells,rbndx)

            vntr = vtndx.size

            print \
                " |-| IT %3i, TNTR %i, VNTR %i, BYP %i"\
                % (it,tntr,vntr,tntr-vntr)

            if ( vntr != tntr ):
                tcrd = tcrd[0:vntr,:]
                tntr = vntr
                # Need to back propogate the rejected tracers.
                for e in range(et):
                    for i in range( len(thlst[e]) ):
                        thlst[e][i] = thlst[e][i][vtndx]

                for i in range(it):
                    ithlst[i] = ithlst[i][vtndx]

            ithlst.append(tcrd.copy()*cellsz +origin)

        thlst.append(ithlst)
        
    print " | EXITING: pthtrace"
    return thlst


#################################################################
#
def pthwrite(
    xthlst, ofpath, desc
    ):
    """
    ----
    
    xthlst          # Extended tracer list.
    ofpath          # Output file path.
    desc            # Short, single-line description of data.

    ----

    Writes results from pthtrace() to VTK file(s) for use with
    ParaView.  Paths need to be stored as vtkPolyLines so that
    the ParaView Tube Filter will surface them correctly.  As
    a result, pthwrite() will generate a VTK file for each Epoch.

    xthlst is an 'extended' thlst from pthtrace() that permits
    external ParaView processing of multiple tracer sets from
    pthtrace().  Each tracer set is assigned a distinct ID in a
    similar fashion to the composition from psvtrace().

    xthlst has the following format:

        [[ID0,thlst0],[ID1,thlst1],...]

    where thlst* are the thlst objects returned directly from
    pthtrace(), and ID* is an ID of the user's choosing.

    NOTE: If only one ID (or tracer set) is needed, then xthlst must 
    take the form
    
        [[ID0,thlst0]].

    pthwrite() makes no constraint on the number of tracers per ID,
    however, the number of Epochs and timesteps must be the same
    across all ID's (ie, pthtrace() must be called with the same 
    tssdiv argument for all ID's).

    NOTE: desc must not contain newline characters ("\n").

    pthwrite() doesn't return anything.

    """
    # Initialization.
    # xthlst[ID][1][Epoch][timestep]
    nids   = len(xthlst)
    nepcs  = len(xthlst[0][1])
    ntstps = len(xthlst[0][1][1])

    ext = ofpath[(len(ofpath)-4):len(ofpath)]
    ext = ext.lower()
    if ( ext != ".vtk" ):
        bofpath = ofpath
        ext = ".vtk"
    else:
        bofpath = ofpath[0:ofpath.rfind(ext)]

    # Get number of points per Epoch (E>0), number of points in E0, and
    # the total number of paths.
    nptspe = 0
    nptse0 = 0
    tnpths = 0
    for id in range(nids):
        npths  = xthlst[id][1][0][0].shape[0]
        nptspe = nptspe +ntstps*npths   # Across all ID's.
        nptse0 = nptse0 +npths          # Across all ID's.
        tnpths = tnpths +npths          # Across all ID's.
        
    # Write the data.
    for e in range(nepcs):
        fh = open('%s-E%03i%s' % (bofpath,e,ext), 'w')

        fh.write('# vtk DataFile Version 3.0\n')

        ostr = '%s (Epoch %i)\n' % (desc,e)
        fh.write(ostr)

        fh.write('ASCII\n')
        fh.write('DATASET POLYDATA\n')
        
        fh.write('POINTS %i double\n' % (e*nptspe +nptse0) )
        for id in range(nids):
            npths = xthlst[id][1][0][0].shape[0]
            for ie in range(e+1):
                if ( ie == 0 ):
                    pntstps = 1
                else:
                    pntstps = ntstps

                for t in range(pntstps):
                    for i in range(npths):
                        fh.write('%e %e %e\n' % \
                                     tuple( xthlst[id][1][ie][t][i,::-1] ))

        if ( e > 0 ):
            ptspl = e*ntstps +1
            lnsp  = 0
            fh.write('LINES %i %i\n' % (tnpths,tnpths*(1+ptspl)) )
            for id in range(nids):
                npths = xthlst[id][1][0][0].shape[0]
                conn  = npths*array( range(e*ntstps +1) ) +lnsp*ptspl
                lnsp  = lnsp +npths
                for p in range(npths):
                    fh.write('%i ' % ptspl)
                    conn.tofile(fh," ","%i")
                    fh.write("\n")
                    conn = conn +1

        fh.write('POINT_DATA %i\n' % (e*nptspe +nptse0))
        fh.write('SCALARS PARTICLEID double\n')
        fh.write('LOOKUP_TABLE default\n')

        for id in range(nids):
            npths   = xthlst[id][1][0][0].shape[0]
            nptspid = (e*ntstps +1)*npths
            idv     = xthlst[id][0]

            idvvec    = empty(nptspid,dtype=float)
            idvvec[:] = idv
            idvvec.tofile(fh,"\n","%f")
            fh.write("\n")
                
        fh.close()


#################################################################
#
def svcinterp( var, crds ):
    """
    ----

    var             # Variable to be interpolated.
    crds            # lx3 array of interpolation coordinates.

    ----

    Spatially interpolates a variable, var, at coordinates, crds,
    using the tricubic scheme of Lekien:2005.

    var can be a PIVVar containing any number of components,
    or a three-dimensional variable containing a scalar.  svcinterp
    treats var as a cell-centered array with data at cell centers.

    crds is an lx3 array or list of coordinates at which var should be
    interpolated.  crds should be expressed in terms of cell indices
    (ie, not mm).  crds must be ordered as follows:
        crds[:,0] -- z-coordinates
        crds[:,1] -- y-coordinates
        crds[:,2] -- x-coordinates

    Returns ivar, an lxn array containing the interpolated variable,
    where n is the number of components in var.
    """
    coeff = bldsvcicoeff(var)
    return flotracec.svcinterp(crds,coeff)


#################################################################
#
def svinterp( var, crds ):
    """
    ----

    var             # Variable to be interpolated.
    crds            # lx3 array of interpolation coordinates.

    ----

    Spatially interpolates a variable, var, at coordinates, crds,
    using trilinear interpolation.

    var can be a PIVVar containing any number of components,
    or a three-dimensional variable containing a scalar.  svinterp
    treats var as a cell-centered array with data at cell centers.

    crds is an lx3 array or list of coordinates at which var should be
    interpolated.  crds should be expressed in terms of cell indices
    (ie, not mm).  crds must be ordered as follows:
        crds[:,0] -- z-coordinates 
        crds[:,1] -- y-coordinates 
        crds[:,2] -- x-coordinates 

    Returns ivar, an lxn array containing the interpolated variable,
    where n is the number of components in var.
    """
    return flotracec.svinterp(var,crds)

    
#################################################################
#
def _splrep(x,y,s):
    """
    ----

    x              # splrep() x argument.
    y              # splrep() y argument.
    s              # splrep() s argument (the smoothness).

    ----

    Helper function for spline interpolation.  _splrep() is a wrapper
    around Scipy's splrep() function that can be called using map().

    Returns spl, the spline object.
    """
    spl = interpolate.splrep(x,y,s=s)
    return spl


#################################################################
#
def _splev(x,spl):
    """
    ----

    x              # splev() x argument.
    spl            # splev() tck argument (the spline object).

    ----

    Helper function for spline interpolation.  _splev() is a wrapper
    around Scipy's splev() function that can be called using map().

    Returns val, the interpolated value.
    """
    val = interpolate.splev(x,spl)
    return val


#################################################################
#
def tcsvar(vlist,tvec,varnm=None):
    """
    ----

    vlist          # List or PIVData object of variables to be interpolated.
    tvec           # List of time values for each entry in vlist.
    varnm=None     # Name of variable if vlist is a PIVData object.

    ----

    Temporally interpolates the variable in vlist using a cubic
    spline.  Each entry in vlist should be the value of the
    variable at the corresponding time given in tvec.

    The length of vlist must match that of tvec.  No check is
    performed.

    Returns tcspl, a dictionary containing:
       'spl' ---- List of spline objects.
       'vshape' - Shape of interpolated variable.
    tcspl can be passed to tcseval() to evaluate the variable
    at a given time.
    """
    # Initialization.
    if ( isinstance(vlist,pivdata.PIVData) ):
        tvlist = []
        for e in vlist:
            tvlist.append( e[varnm] )
        vlist = tvlist
    
    vshape = array(vlist[0].shape)
    tnval  = vshape.prod()

    ntime = len(tvec)
    tvec  = array(tvec).reshape((1,ntime)).repeat(tnval,0)

    vlist = array(vlist).reshape(ntime,tnval).transpose()

    # Smoothing vector for splrep().  Set to zero for interpolation.
    svec = zeros(tnval)

    # Get the splines.
    spl = map(_splrep,tvec,vlist,svec)

    tcspl = {'spl':spl,'vshape':vshape}

    return tcspl


#################################################################
#
def tcseval(tcspl,tval):
    """
    ----

    tcspl          # Dictionary from tcsvar.
    tval           # Time value at which to evaluate the variable.

    ----

    Evaluates an interpolated variable at tval.

    Returns ivar, the variable evaluated at tval.
    """
    # Initialization.
    spl    = tcspl['spl']
    vshape = tcspl['vshape']
    tnval  = vshape.prod()

    tvec    = empty(tnval,dtype=float)
    tvec[:] = tval

    # Evaluate the variable.
    ivar = flotracec.tcseval(spl,tvec)
    ivar = ivar.reshape(vshape)

    return ivar


#################################################################
#
def velfun( tcrd, st, fargs ):
    """
    ----

    tcrd            lx3 array of tracer coordinates.
    st              Incremental step time from RK scheme.
    fargs           List of additional required variables.

    ----

    Runge-Kutta RHS function that computes velocity at position
    tcrd and time t + st.

    Let t represent the current time which lies between two Epochs,
    t(i) represent the timestamp for a particular Epoch, and dt
    equal the time between Epochs, then fargs is a list containing:
        fargs[0] ---- velocity field at time t(i).
        fargs[1] ---- velocity field at time t(i) +dt.
        fargs[2] ---- t(i)
        fargs[3] ---- rt, where rt = t -t(i).
        fargs[4] ---- dt, where dt = t(i+1) -t(i).
        fargs[5] ---- tcspl dictionary or None.
        fargs[6] ---- True for tricubic spatial interpolation.

    fargs[5] and fargs[6] control the type of interpolation used
    for time and space, respectively.
    - If fargs[5] is set to a tcspl dictionary from tcsvar(), then
      cubic interpolation through time will be used.  Set
      fargs[5] = None to use linear temporal interpolation.
    - If fargs[6] is set to True, tricubic interpolation will be
      used for spatial variables.  Set fargs[6] = False for
      trilinear spatial interpolation.

    Returns tvel, a lx3 array of velocity vectors.
    """
    # Initialization.
    srt  = fargs[3] +st
    asrt = abs(srt)

    eps = finfo(float).eps

    # Temporally interpolate velocity field.  If asrt < eps, then we
    # don't need to interpolate in time.
    if ( asrt > fargs[2]*eps ):
        if ( compat.checkNone(fargs[5]) ):
            ft   = (srt)/fargs[4]
            ivel = (1. -ft)*fargs[0] +ft*fargs[1]
        else:
            cst  = fargs[2] +srt
            ivel = tcseval(fargs[5],cst)
    else:
        ivel = fargs[0]

    # Spatially interpolate velocity field.
    if ( fargs[6] == True ):
        tvel = svcinterp(ivel,tcrd)
    else:
        tvel = svinterp(ivel,tcrd)

    return tvel

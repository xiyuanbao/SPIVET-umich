"""
Filename:  pivdata.py
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
  Module containing PivLIB data classes and worker functions
  used by the high level PivLIB drivers.
  
  NOTE: If a the amount of PIV data that is being processed 
  exceeds a gigabyte, it is in the user's interest to get the 
  data into PIVEpoch's as soon as possible.  Otherwise the
  process may run out of addressable memory.  See PIVEpoch
  documentation for more details.  For more information on the
  cache mechanisms used by SPIVET, see the code.

Contents:
  cpivvar()
  getdtype()
  getex2vdata()
  importex2vars()
  loadpivdata()
  prsex2vname()  

  class PIVCache
  class PVStore
  class PIVVar
  class PIVEpoch
  class PIVData
"""

from numpy import *
from spivet import spivetrev
import pivpickle, pivlibc
import exodusII as ex2
import datetime, pickle, tempfile, sys, os, stat, shutil, uuid
from spivet import compat

"""
*** SPIVET Cache Mechanism Overview ***

Although the migration to full 64-bit OS's and applications is underway, 
most Python builds (as of Python 2.5) are still 32-bit for various reasons.  
For very large datasets, the address limit for a 32-bit application can
easily be reached.  Once the max amount of virtual memory allocated to
a particular process has been exhausted (~2-3 GB for a 32-bit app on most 
OS's), a request for additional memory will cause the application to 
crash.  To ease the memory footprint of SPIVET applications, caching 
mechanisms are employed.

The smallest chunk of data that is currently considered for any cache
operation is the PIVVar.  That is PIVVar's in their entirety are
stored in one of the caches.  Until PIVVar's are added to a PIVEpoch
object, all of the PIVVar objects are 'active' and stored in virtual
memory (either RAM or OS swap, but within the confines of the address
space available to the process).  As previously noted, however, storing
all PIVVar's in process memory may cause issues.  

Once PIVVar's are added to a PIVEpoch, a PIVCache object manages which 
variables are 'active' and which are 'deactivated.'  The set of active  
variables maintained by each Epoch constitutes one of the SPIVET caches. 
Note: In order for any operations to be performed on a variable, it must
be active.  PIVEpochs that are further stored in a PIVData object 
relinquish management of the active variable cache to the PIVCache of 
the PIVData object.  Having the PIVData object service active variables 
from a cache shared among all contained Epochs provides a significant
reduction in memory footprint.  The PIVData object's csize parameter 
will override that of all contained PIVEpochs.

After the number of variables in a PIVCache exceeds the csize parameter, 
the PIVCache object will signal the parent PIVEpoch to store stale variables
to a disk cache based on when the variable was last used (oldest variables 
get cached to disk to first).  The disk-based caching mechanism consists 
of a single file for each deactivated PIVVar.

One other disk-based cache mechanism for PIVVar's is employed via the
ExodusII file.  When a PIVData object is loaded from an ExodusII file,
the variable data within the ExodusII file is not immediately loaded.  
Each PIVVar will be activated from the ExodusII file only when the 
variable is accessed.  Once activated, a future deactivation of a PIVVar
will be done to a new file as described in the previous paragraph (ie, 
the EX2 file is read only).

In summary, three caches are used:
    1) Cache of active variables stored in memory.  This cache is
       managed by a PIVCache object.
    2) Disk-based storage of individual PIVVar objects.  Each file
       is writeable only once.
    3) Upon loading an ExodusII file, the PIVVar objects are only
       activated from the EX2 file on an as-needed basis.  The EX2
       cache is read only.
"""


#################################################################
#
def cpivvar(data,name="NOTSET",units="NOTSET",vtype='E'):
    """
    ----

    data            # Input data array.
    name            # ID name of data.
    units           # Text describing data units.
    vtype           # Flag specifying that data is cell-centered or nodal.
    
    ----

    Creates a PIVVar from numpy array data.  

    data must be of appropriate shape, which means data must have a 
    minimum of three dimensions and a maximum of 4.  See PIVVar class
    documentation for more details.

    Returns var, a PIVVar object containing the data in data.  Note that
    a copy is made of the data.
    """
    # Initialization.
    shape = data.shape
    dims  = data.ndim

    data = array(data)

    if ( dims == 3 ):
        data  = data.reshape((1,shape[0],shape[1],shape[2]))
        shape = data.shape
    elif ( dims != 4 ):
        raise ValueError,\
            "data must have 3 or 4 dimensions."

    return PIVVar(shape,
                  name=name,
                  units=units,
                  dtype=data.dtype,
                  vtype=vtype,
                  data=data)


#################################################################
#
def getdtype( name, info ):
    """
    ----

    name            # Variable name.
    info            # List of info strings from ExodusII file.

    ----

    Extracts the variable dtype from an info string if available or
    assigns a default dtype of float.
    """
    for i in range( len(info) ):
        kv = info[i].split(':',1)

        if ( len(kv) != 2 ):
            continue

        key = kv[0]
        val = kv[1]
        
        kf = key.split('.',1)

        if ( ( len(kf) != 2 ) 
             or ( kf[0] != name ) 
             or ( kf[1] != "DTYPE" ) ):
            continue
        
        return dtype(val.strip())

    # A valid DTYPE entry wasn't found in the info list.
    return dtype(float)

#################################################################
#
def getex2vdata( exo, tstp, vtype, vndx ):
    """
    ----
    
    exo             # The ExodusII file object.
    tstp            # Time step (ie, Epoch + 1).
    vtype           # Variable type ("E" or "N").
    vndx            # Variable index (starts at 1).

    ----

    Loads the data from an ExodusII variable of index vndx and
    type vtype at time step tstp.

    Returns data, a 1D numpy array containg the data.

    """
    if ( vtype.upper() == "N" ):
        return ex2.ex_get_nodal_var(exo,tstp,vndx)
    elif ( vtype.upper() == "E" ):
        return ex2.ex_get_elem_var(exo,tstp,vndx,1)
    else:
        print "ERROR: Unsupported vtype %s" % vtype
        return None


#################################################################
#
def importex2vars( exo, tstp, vtype, ncells ):
    """
    ----

    exo             # ExodusII file object.
    tstp            # Time step (ie, Epoch + 1).
    vtype           # Variable type ("E" or "N").
    ncells          # Number of cells (nz, ny, nx).

    ----

    Returns a list of PVStore objects for the PIVVar's of type
    vtype at time step tstp.

    NOTE:  Multi-component variables (eg, vectors) are expected
    to be stored in the ExodusII file in the following order
        Vector ---- X, Y, Z
        Tensor ---- XX, XY, XZ, YX, YY, YZ, ZX, ZY, ZZ
    """
    # Initialization.
    ncells = array(ncells)

    info = ex2.ex_get_info(exo)
    vnms = ex2.ex_get_var_names(exo,vtype)

    # Determine number of variables, their names, and dimensionality.
    vvars = []
    i     = 0
    while ( i < len(vnms) ):
        [name,units,cmp] = prsex2vname(vnms[i])
        dtype            = getdtype(name,info)

        if ( len(cmp) == 0 ):
            ncmp = 1
        elif ( len(cmp) == 1 ):
            ncmp = 3
        elif ( len(cmp) == 2 ):
            ncmp = 9
        else:
            raise ValueError("Unknown number of variable components: %i" % \
                             ncmp)

        vndx = arange(ncmp,dtype=int) +i +1
        vndx = vndx[::-1]
        
        if ( vtype == 'N' ):
            shape = array([ncmp,ncells[0]+1,ncells[1]+1,ncells[2]+1])
        else:
            shape = array([ncmp,ncells[0],ncells[1],ncells[2]])
        
        pvs = PVStore(exo,shape,name,units,dtype,vtype,'X',tstp,vndx)

        i = i +ncmp
        vvars.append(pvs)

    return vvars


#################################################################
#
def loadpivdata( ifpath ):
    """
    ----

    ifpath          # Input file path.

    ----

    Loads a PIVData object from an ex2 file.  
    """
    # Initialization.
    try:
        exo = ex2.ex_open(ifpath,ex2.exc["EX_READ"],8)
    except:
        return pivpickle.pklload(ifpath)
    
    # Determine if this is a valid PIVData file.
    info  = ex2.ex_get_info(exo)
    valid = False
    for i in range(len(info)):
        if ( info[i] == "PIVDATAOBJECT" ):
            valid = True
            break
        elif ( info[i][0:12] == "TIME UNITS [" ):
            valid = True
            break

    if ( not valid):
        print "ERROR: Not a valid PIVData file."
        return None

    # Get time units.
    for i in range(len(info)):
        if ( info[i][0:12] == "TIME UNITS [" ):
            [dmy,tunits] = info[i].rsplit("[",1)
            [tunits,dmy] = tunits.rsplit("]",1)
            
            break

    # Determine if time is stored in a global variable.
    ngv = ex2.ex_get_var_param(exo,"G")

    gvtndx = -1
    if ( ngv > 0 ):
        gvnms = ex2.ex_get_var_names(exo,"G")
        for i in range( len(gvnms) ):
            if ( gvnms[i] == "EPOCHTIME" ):
                gvtndx = i 
                break

    # Try to get ncells, cellsz, and origin directly.
    try:
        nzc = ex2.ex_get_prop(exo,ex2.exc["EX_ELEM_BLOCK"],1,"NZCELLS")
        nyc = ex2.ex_get_prop(exo,ex2.exc["EX_ELEM_BLOCK"],1,"NYCELLS")
        nxc = ex2.ex_get_prop(exo,ex2.exc["EX_ELEM_BLOCK"],1,"NXCELLS")

        for i in range(len(info)):
            kv = info[i].split(':',1)

            if ( len(kv) != 2 ):
                continue

            key = kv[0]
            val = kv[1]
            if ( key == "PIVDATA-ZORIGIN"):
                zorg = pickle.loads(val)
            elif ( key == "PIVDATA-YORIGIN"):
                yorg = pickle.loads(val)
            elif ( key == "PIVDATA-XORIGIN"):
                xorg = pickle.loads(val)
            elif ( key == "PIVDATA-ZCELLSZ"):
                zcsz = pickle.loads(val)
            elif ( key == "PIVDATA-YCELLSZ"):
                ycsz = pickle.loads(val)
            elif ( key == "PIVDATA-XCELLSZ"):
                xcsz = pickle.loads(val)

        ncells = array([nzc,nyc,nxc])
        cellsz = array([zcsz,ycsz,xcsz])
        origin = array([zorg,yorg,xorg])
    except:
        # Need to compute ncells, cellsz, and origin.
        ncoord = ex2.ex_get_coord(exo)
        econn  = ex2.ex_get_elem_conn(exo,1)

        e0nds = econn[0:8] -1

        nxnds = econn[3] -econn[0]
        nynds = (econn[4] -econn[0])/nxnds
        nznds = ncoord[0].size/(nxnds*nynds)
        
        ncells = [nznds -1, nynds -1, nxnds -1]
        ncells = array(ncells)

        e0xnc = ncoord[0][e0nds]
        e0ync = ncoord[1][e0nds]
        e0znc = ncoord[2][e0nds] 
    
        cellsz = [e0znc[4] -e0znc[0], e0ync[2] -e0ync[0], e0xnc[1] -e0xnc[0]]
        cellsz = array(cellsz)

        origin = array([e0znc[0],e0ync[0],e0xnc[0]]) +cellsz/2.

    # Get description and initialize the PIVData object.
    rval = ex2.ex_inquire(exo,ex2.exc["EX_INQ_TITLE"])

    pd = PIVData(cellsz,origin,rval[2],exo=exo)

    # Get the QA records.
    qa = ex2.ex_get_qa(exo)
    for i in range( len(qa) ):
        pd.addQA(qa[i])

    # Fill the variables.
    rval   = ex2.ex_inquire(exo,ex2.exc["EX_INQ_TIME"])
    ntstps = rval[0]

    for e in range(ntstps):
        vars = []
        vars.extend( importex2vars(exo,e+1,"N",ncells) )
        vars.extend( importex2vars(exo,e+1,"E",ncells) )

        if ( gvtndx > -1 ):
            gvars = ex2.ex_get_glob_vars(exo,e+1)
            time  = gvars[gvtndx]
        else:
            time = ex2.ex_get_time(exo,e+1)

        pd.addVars(e,vars)
        pd[e].setTime(time,tunits)

    # Don't close the file here.  That is handled by PIVData object.
    
    return pd
    

#################################################################
#
def prsex2vname( name ):
    """
    ----

    name            # ExodusII variable name.

    ----

    Takes an ExodusII variable name from ex_get_var_names() and
    parses it.  

    Exodus II variable names are expected to have the following
    format:
        "name [units] cmp"

    Returns [name,units,cmp], where
        name ---- The variable name.
        units --- The variable units.
        cmp ----- The variable component (eg, Z).
    """

    [name,units] = name.rsplit("[",1)
    [units,cmp]  = units.rsplit("]",1)
    
    name  = name.strip()
    units = units.strip()
    cmp   = cmp.strip()

    return [name,units,cmp]


#################################################################
#
class PIVCache(dict):
    """
    A PIVCache manages a collection of objects stored in memory in an
    effort to maximize efficient virtual memory utilization.  Cached
    objects that are stored in the PIVCache are deemed 'active' objects.
    Once the number of objects in the cache exceeds the csize parameter,
    'stale' objects will be removed (ie, deactivated) from the cache on 
    a least recently used basis thereby freeing up room for more recently 
    used objects.

    The PIVCache permits associating a callback function with each stored 
    object.  When a stale variable is deactivated from the cache, the 
    callback function will be called so that appropriate action can
    be taken.  The callback must have the signature
            callback(ouuid,obj)
    where ouuid is the unique identifier that was used to refer to the
    previously cached object in the PIVCache, and obj is a reference
    to the object itself.
    """
    def __init__(self,csize=6):
        list.__setattr__(self,'m_residents',[]) # Cache residents.
        self.setCacheSize(csize)

    def __getattr__(self,name):
        if ( name == 'csize' ):
            return self.m_csize           # Target cache size.
        elif ( name == 'ccsize' ):
            return len(self.keys())       # Current cache size.
        else:
            raise AttributeError,\
                "PIVCache has no attribute " + name

    def __getitem__(self,ouuid):
        """
        Internal mechanism for retrieving values.  Do not call __getitem__()
        directly.  Use retrieve() or the bracket [] notation instead.
        """
        return self.retrieve(ouuid)

    def assimilateCache(self,acache):
        """
        Migrates the contents of another PIVCache, acache, into this cache.
        Note that assimilateCache() will remove the contents of acache
        (ie, acache is modified).  After a call to assimilateCache(),
        acache can be discarded (it will point to an empty dictionary).
        """
        # Don't assimilate self.
        if ( id(self) == id(acache) ):
            raise ValueError("Cannot assimilate self.")
            
        for k in acache.keys():
            [obj,cbfun] = acache.pop(k)
            self.store(obj,cbfun,k)

    def retrieve(self,ouuid):
        """
        Retrieve the object corresponding to the UUID ouuid from the
        cache.  Retrieved object will be classified as most recently
        used.  
        
        NOTE:  retrieve() does not remove the object from the cache.
        
        Returns obj, a reference to the cached object.
        """
        if ( self.has_key(ouuid) ):
            self.store(ouuid)
            return dict.__getitem__(self,ouuid)[0]
        else:
            raise KeyError(str(ouuid))

    def setCacheSize(self,csize):
        """
        Sets the cache size (ie, the number of items that are stored in 
        virtual memory).  
        """
        if ( csize == 0 ):
            raise ValueError("csize must be greater than zero.")
 
        list.__setattr__(self,'m_csize',csize)

    def store(self,obj,cbfun=None,ouuid=None):
        """
        Adds obj to cache of objects stored. obj can be either a
        valid uuid of an existing cached object (in which case, the
        object will be marked most recently used) or any other object
        to be cached. 
        
        cbfun is a callback function that should be called if obj becomes
        stale (see below).    
        
        ouuid is a unique identifier for the object stored in the cache.
        If ouuid = None, store() will automatically generate a UUID for 
        the object and return the generated value to the caller.  Generally 
        speaking, the user should permit store() to generate a UUID (ie,
        leave ouuid = None).  If store() is called with ouuid set to a
        value other than None, and the cache already contains an object
        associated with that ouuid, then the currently cached object
        will be replaced with the contents of obj.
        
        A stale object is an object that has not been recently used and
        must be removed from the cache to make room for other active 
        objects.  When an object becomes stale, cbfun will be called.
        cbfun must have the signature
            cbfun(ouuid,obj).
        
        Returns ouuid, a UUID for the cached object.  
        """
        res  = self.m_residents
        tcsz = self.m_csize

        # Remove any dangling ouuid's from the residents list.  The
        # residents list can get out of sync with the dictionary if
        # the user calls pop() or related methods on the dictionary.
        if ( len(res) > len(self.keys()) ):
            cpos = 0
            while ( cpos < len(self.keys()) ):
                if ( not self.has_key(res[cpos]) ):
                    res.pop(cpos)
                    continue
                else:
                    cpos = cpos +1
        elif ( len(res) < len(self.keys()) ):
            # This situation shouldn't occur.
            for k in self.keys():
                try:
                    ndx = res.index(k)
                except ValueError:
                    res.append(k)
                
        # Get ouuid and add obj to the dictionary if necessary.
        if ( not isinstance(obj,uuid.UUID) ):
            if ( compat.checkNone(ouuid) ):
                ouuid = uuid.uuid4()
                #print "CREATED UUID %s" % str(ouuid) ##### DEBUG
                # Ensure the ouuid is unique.
                while (True):
                    if ( self.has_key(ouuid) ):
                        ouuid = uuid.uuid4()
                        continue
                    else:
                        break
                
            dict.__setitem__(self,ouuid,[obj,cbfun])
        else:
            ouuid = obj

        # Deactivate stale objects.
        cpos = 0
        ccsz = len( res )
        while ( cpos < ccsz ):
            if ( ccsz < tcsz  ):
                break
            
            # If a resident object is the key object, we'll handle
            # that later.
            vuuid = res[cpos]
            if ( vuuid == ouuid ):
                cpos = cpos +1
                continue
            
            # Any object in the residents list should be in the 
            # dictionary, so this is safe.
            [obj,cbfun] = dict.__getitem__(self,vuuid)

            # The following should be the minimum references:
            #   (1) PIVCache
            #   (2) obj above
            #   (3) The reference taken in the call to getrefcount()
            if ( sys.getrefcount(obj) < 4 ):
                #print "DEACTIVATING %s, RCNT %i" % (vuuid,sys.getrefcount(obj)) ##### DEBUG                
                if ( not compat.checkNone(cbfun) ):
                    # Need to pass object back to cbfun since PIVCache
                    # has the only copy.
                    cbfun(vuuid,obj)
                self.pop( res.pop(cpos) )
                ccsz = ccsz -1
                continue

            cpos = cpos +1

        # Mark the object most-recently-used by moving it to the
        # end of the residents list.
        try:
            ndx = res.index(ouuid)
            res.pop(ndx)
        except ValueError:
            pass
                    
        res.append(ouuid)
        return ouuid


#################################################################
#
class PVStore():
    """
    Utility class that represents a deactivated PIVVar.  Intended
    primarily for use by PIVEpoch caching. 
    """
    def __init__(self,fh,shape,name,units,dtype,vtype,source="T",tstp=None,vndx=None):
        """
        ----
        
        fh         # Path to temporary file or ExodusII file handle.
        shape      # PIVVar shape.
        name       # Variable name.
        units      # Variable units.
        dtype      # Variable dtype.
        vtype      # Variable vtype.
        source="T" # File source.  T - temp file, X - ExodusII file.
        tstp=None  # Timestep into the ExodusII file.
        vndx=None  # List of variable indices into the ExodusII file.

        ----
        """
        self.fh     = fh
        self.shape  = shape
        self.name   = name
        self.units  = units
        self.dtype  = dtype
        self.vtype  = vtype
        self.source = source
        self.tstp   = tstp
        self.vndx   = vndx

    def __del__(self):
        """
        Cleanup any temp file lying around.
        """
        source = self.source.upper()        
        if ( source == 'T' and os.path.exists(self.fh) ):
            os.remove(self.fh)

    def activate(self):
        """
        Reconstitutes the deactivated PIVVar from disk.
        """
        source = self.source.upper()
        if ( source == 'T' ):
            tpath = self.fh
            fh    = open(tpath,'rb')
            data  = fromfile(fh,dtype=self.dtype)
            fh.close()
            os.remove(tpath)
            data = data.reshape(self.shape)
        elif ( source == 'X' ):
            exo   = self.fh
            shape = self.shape
            vtype = self.vtype
            tstp  = self.tstp
            vndx  = self.vndx

            ncmp = shape[0]
            data = range(ncmp)
            for c in range(ncmp):
                data[c] = getex2vdata(exo,tstp,vtype,vndx[c])
    
            data = array(data).astype(self.dtype)
            data = data.reshape(shape)
        else:
            raise ValueError("Invalid source %s" % source)
                
        pv = cpivvar(data,name=self.name,units=self.units,vtype=self.vtype)
        return pv


#################################################################
#
class PIVVar(ndarray):
    """
    PivLIB high-level driver functions like ofcomp() and tfcomp() simply
    compute an array of velocity or temperature vectors with some
    regular spacing between values.  For further processing of PivLIB data,
    it is convenient to interpret the PivLIB results as lying on a grid
    (similar to what would be used in a CFD computation), where space is
    discretized into a number of hexahedral cells with nodes at the vertices
    of the cells.  
    
    PIVVar objects provide the fundamental data class for storing PivLIB 
    results in a grid-like manner.  A PIVVar object is a subclass of NumPy's
    ndarray that enforces some shape constraints and stores some additional
    descriptive attributes.

    PIVVar stores data as an ncmp x nz x ny x nx array, where nz, ny, and nx 
    represent the number of data points in the z, y, and x directions, 
    respectively.  ncmp represents the number of data components (ie, 
    whether the data is a tensor, vector, or a scalar). As an example, a 
    PIVVar object would be used to store the displacement values computed 
    with ofcomp().  Displacements are vectors in 3-space, so ncmp = 3 in 
    this case.

    The data points stored by a PIVVar can correspond to flow variables 
    at cell centers or nodes depending on the user's needs.  PIVVar 
    provides a flag, vtype, that allows the user to specify which.

    Because a PIVVar is a ndarray, all arithmetic operations used on 
    ndarrays can be done with PIVVars.  Take for example b = a + 2, where
    a is a PIVVar.  In this case, b will also be a PIVVar, but it will
    not inherit a's name or units.

    NOTE: To be consistent with PivLIB's coordinate system, tensors should 
    be stored as 
       zz zy zx yz yy yx xz xy xx 
    """
    def __new__(subtype,shape,name="NOTSET",units="NOTSET",dtype=float,
                vtype='E',data=None):
        """
        A PIVVar object must be initialized with:
            shape ---- /Int/ (ncmp,nz,ny,nx) dimensions of the array.  ncmp
                       can be one of:
                           1 ---- Scalar
                           3 ---- Vector
                           9 ---- Tensor

        The following parameters are optional:
            name ----- /String/ ID name of the data (eg, OFDELTAY).  name
                       will be truncated to a max length of 31 characters.
            units ---- /String/ Text describing the units (eg, MM_PIX). units
                       will be truncated to a max length of 31 characters.
            dtype ---- /Int/ The desired data-type for the array.  Defaults
                       to double precision float.
            vtype ---- /String/ The type of data stored in the array.  vtype
                       can be one of two values:
                           E --- The variable contains cell-centered data.
                           N --- The variable contains nodal data.
            data ----- /Array/ Data to be stored in the PIVVar.

        NOTE:  The returned array will be initialized to zero if data = None,
        else a copy of the data will be made.
        """
        shape = array(shape)
        if ( shape.size != 4 ):
            raise ValueError,\
                "PIVVar shape must be a list/array of length 4."
        if ( ( shape[0] != 1 ) and ( shape[0] != 3 ) and ( shape[0] != 9 ) ):
            raise ValueError,\
                "PIVVar shape must specify an ncmp of 1, 3, or 9."
        
        # Pass initialization parameters to the parent class.  Note:
        # ndarray is a bit wierd with subclassing.  It uses __new__()
        # and __array_finalize__() instead of the traditional Python
        # __init__().
        obj = ndarray.__new__(subtype,shape,dtype)

        # Setup attributes.
        ndarray.__setattr__(obj,'m_max_str_len',ex2.EX_MAX_STR_LENGTH -1)
        obj.setName(name)
        obj.setUnits(units)
        obj.setVtype(vtype)

        if ( not compat.checkNone(data) ):
            obj[:,:,:,:] = data.reshape(shape)
        else:
            obj[:,:,:,:] = 0.

        return obj

    def __array_finalize__(self,obj):
        # __new__() doesn't get called for every array operation (eg,
        # with a statement like c = a), but __array_finalize__() does.
        # If the attributes aren't set to default values here, then 
        # derived objects will not behave correctly.
        #
        # ndarray methods (eg, std()) return an array of subclass type.
        # Further operations on the result can generate errors unless
        # it is fully equipped as a PIVVar.  Set the vtype of such
        # results to "X".
        ndarray.__setattr__(self,'m_name','NOTSET')
        ndarray.__setattr__(self,'m_units','NOTSET')
        ndarray.__setattr__(self,'m_max_str_len',ex2.EX_MAX_STR_LENGTH -1)
        if ( isinstance(obj,PIVVar) ):
            ndarray.__setattr__(self,'m_vtype',obj.m_vtype)
        else:
            ndarray.__setattr__(self,'m_vtype','X')
            
    def __deactivate__(self):
        """
        Writes array contents to a temporary file.  Mainly intended to
        be used with PIVEpoch variable caching.

        Returns a PVStore object.
        """
        [fd,fp] = tempfile.mkstemp()
        fh = os.fdopen(fd,'wb')
        ndarray.tofile(self,fh)
        fh.flush()
        fh.close()
        return PVStore(fp,
                self.shape,
                self.m_name,
                self.m_units,
                self.dtype,
                self.m_vtype,
                'T')

    def __getattr__(self,name):
        if ( name == 'name' ):
            return self.m_name
        elif ( name == 'units' ):
            return self.m_units
        elif ( name == 'vtype' ):
            return self.m_vtype
        else:
            raise AttributeError,\
                "PIVVar has no attribute " + name

    def __getitem__(self,y):
        # Any slices should be returned as type ndarray since they
        # will no longer have the correct shape.

        # Get the slice.
        itm = ndarray.__getitem__(self,y)
        return asarray(itm)

    def __reduce__(self):
        # Called before pickling.
        sprred   = list(ndarray.__reduce__(self))
        slfstate = (self.m_max_str_len,
                    self.m_name,
                    self.m_units,
                    self.m_vtype)
        sprred[2] = (sprred[2],slfstate)
        return tuple(sprred)

    def __setstate__(self,state):
        # Called when unpickling.
        sprstate = state[0]
        slfstate = state[1]

        ndarray.__setstate__(self,sprstate)

        ndarray.__setattr__(self,"m_max_str_len",slfstate[0])
        ndarray.__setattr__(self,"m_name",slfstate[1])
        ndarray.__setattr__(self,"m_units",slfstate[2])
        ndarray.__setattr__(self,"m_vtype",slfstate[3])

    def __setattr__(self,name,value):
        raise AttributeError,\
            "Please use methods to set PIVVar attributes."

    def copy(self):
        # A copy should return an array of type PIVVar.
        return PIVVar(self.shape,
                      name=self.name,
                      units=self.units,
                      dtype=self.dtype,
                      vtype=self.vtype,
                      data=self)

    def printStats(self):
        """
        Prints a statistical summary of data in the array.
        """
        print "----- %s [%s] -----" % (self.m_name,self.m_units)
        if ( self.shape[0] == 3 ):
            print "      Z             Y             X"
            print "MIN  %13e %13e %13e" %\
                (self[0].min(),self[1].min(),self[2].min())
            print "MAX  %13e %13e %13e" %\
                (self[0].max(),self[1].max(),self[2].max())
            print "MEAN %13e %13e %13e" %\
                (self[0].mean(),self[1].mean(),self[2].mean())
            print "STD  %13e %13e %13e" %\
                (self[0].std(),self[1].std(),self[2].std())
        else:
            print "MIN  %13e" % self.min()
            print "MAX  %13e" % self.max()
            print "MEAN %13e" % self.mean()
            print "STD  %13e" % self.std()

    def setAttr(self,name,units,vtype='E'):
        """
        Convenience function to set all descriptive parameters at once.
        """
        self.setName(name)
        self.setUnits(units)
        self.setVtype(vtype)

    def setName(self,name):
        """
        Sets a name for the stored data.  name will be truncated to
        31 characters for compatibility with ExodusII.
        """
        if ( not isinstance(name,str) ):
            raise ValueError,\
                "name must be a string."
        nl = min(len(name),self.m_max_str_len)

        ndarray.__setattr__(self,"m_name",name[0:nl])

    def setUnits(self,units):
        """
        Sets the units string for the stored data.  units will be truncated
        to 31 characters for compatibility with ExodusII.
        """
        if ( not isinstance(units,str) ):
            raise ValueError,\
                "units must be a string."        
        ul = min(len(units),self.m_max_str_len)

        ndarray.__setattr__(self,"m_units",units[0:ul])

    def setVtype(self,vtype):
        """
        Sets the vtype string for the stored data.  vtype can be one
        of two values:
            E --- The variable contains cell-centered data.
            N --- The variable contains nodal data.
        """
        vtype = vtype.upper()
        if ( (vtype != "E") and (vtype != "N") ):
            raise ValueError,\
                "PIVVar vtype must be either 'E' or 'N'."

        ndarray.__setattr__(self,"m_vtype",vtype)

#################################################################
#
# IMPLEMENTATION NOTE: A PIVEpoch does not store PIVVar's directly,
# as that is handled by the m_cache data member.  Instead, a PIVEpoch
# can store either a UUID object or a PVStore object. 
class PIVEpoch(dict):
    """
    PIV data may be collected at different times (epochs) so that transient
    phenomenon can be studied.  The PIVEpoch provides a container for 
    a collection of PIVVar's that are associated with a given time.

    A PIVEpoch object is a subclass of Python's dictionary, so PIV variables
    are stored by and can be retrieved by the variable's name (as well as
    any other valid dictionary access method).

    NOTE:  All PIVVar's in a given epoch must have a unique name.  Storing
           a new variable with the same name as an existing variable will
           cause the existing variable to be silently overwritten.
    NOTE:  All PIVVar's in a given epoch must have the same shape as the
           the other elements of that vtype.  If a PIVVar of vtype E is
           stored in the Epoch, then all other PIVVar's of vtype E must
           have the same (z,y,x) shape (ncmp can vary).  Additionally,
           once the element shape is set, the permissible node count is
           also set for PIVVar's of vtype N.  Specifically, 
           nshape = eshape +1, where nshape is (z,y,x) shape for PIVVar's
           of vtype N.
           
    PIVEpoch's also provide a mechanism for variable caching.  Although the
    migration to full 64-bit OS's and applications is underway, most Python
    builds (as of Python 2.5) are still 32-bit for various reasons.  For
    very large datasets, the address limit for a 32-bit application can
    easily be reached.  Once the max amount of available process virtual
    memory has been exhausted (~2-3 GB for a 32-bit app on most OS's), a
    request for additional memory will cause the application to crash.  To
    ease the memory footprint of PivLIB, the PIVEpoch sports a variable
    cache.  The caching mechanism is transparent to the user.  In fact,
    the only tunable component of the cache is set with the csize parameter
    during PIVEpoch object instantiation.  The parameter csize controls the
    number of variables stored in virtual memory for the PIVEpoch.  All
    other variables in the PIVEpoch will be dumped to disk.  Changing
    csize to a value other than the default should not be necessary.
    """
    def __init__(self,time,units="NOTSET",vars=None,csize=6,cache=None):
        """
        A PIVEpoch object must be initialized with:
            time ----- /Float/ The time value for the epoch.

        Optionally, the following can be specified:
            units ---- /String/ The time units (eg, sec).  units will be 
                       truncated to 31 characters.
            vars ----- /PIVVar/ A single PIVVar or a list of PIVVar's.
            csize ---- /Int/ Cache size.  User should not need to modify
                       csize.
            cache ---- /PIVCache/ A PIVCache object for the Epoch to use.
                       If cache is provided, csize will be ignored.
        """
        list.__setattr__(self,'m_max_str_len',ex2.EX_MAX_STR_LENGTH -1)
        list.__setattr__(self,'m_time',float(time))

        list.__setattr__(self,'m_eshape',None)    # Number of elements.
        list.__setattr__(self,'m_nshape',None)    # Number of nodes.
        
        self.setTime(time,units)

        if ( not compat.checkNone(cache) ):
            self.setCache(cache)
        else:
            self.setCache( PIVCache(csize) )
        
        list.__setattr__(self,'m_residents',[]) # Cache residents.
        
        if ( not compat.checkNone(vars) ):
            self.addVars(vars)

    def __deactivate__(self,ouuid,var):
        """
        Method called by PIVCache to deactivate the variable var.  Users
        should not call this method.
        """
        if ( not self.has_key(var.name) ):
            # We can't call rebuildKeys() as the ouuid stored in the
            # dictionary has been removed from the cache.
            ival = self.iteritems()
            for val in ival:
                if ( isinstance(val[1],uuid.UUID) and val[1] == ouuid ):
                    self.pop(val[0])
                    dict.__setitem__(self,var.name,ouuid)
                    break
            
        # This should now be safe.
        pvs = var.__deactivate__()
        dict.__setitem__(self,var.name,pvs)
        
    def __getattr__(self,name):
        if ( name == 'time' ):
            return self.m_time
        elif ( name == 'units' ):
            return self.m_units
        elif ( name == 'eshape' ):
            return self.m_eshape.copy()
        elif ( name == 'nshape' ):
            return self.m_nshape.copy()
        else:
            raise AttributeError,\
                "PIVEpoch has no attribute " + name
        
    def __setattr__(self,name,value):
        raise AttributeError,\
            "Please use methods to set Epoch attributes."

    def __getitem__(self,key):
        """
        Internal mechanism for retrieving values.  Do not call __getitem__()
        directly.  Use the bracket [] notation instead.
        """
        if ( not self.has_key(key) ):
            self.rebuildKeys()
            if ( not self.has_key(key) ):
                raise KeyError(key)
        
        value = dict.__getitem__(self,key) 
        if ( isinstance(value,PVStore) ):
            var   = value.activate()
            ouuid = self.m_cache.store(var,self.__deactivate__)
            dict.__setitem__(self,key,ouuid)
        else:
            var = self.m_cache.retrieve(value)   
        
        return var
            
    def __setitem__(self,key,value):
        """
        Internal mechanism for storing values.  Do not call __setitem__()
        directly.  Instead, use the bracket [] notation or preferably 
        addVars().
        """
        if ( not isinstance(value,PIVVar) and not isinstance(value,PVStore) ):
            raise ValueError,\
                "PIVEpoch can only store PIVVar or PVStore objects."

        # If the key exists but doesn't match value.name, raise an error.
        if ( key != value.name ):
            raise KeyError,\
                "Dictionary key must match PIVVar.name"

        # Check that the shape of the var is compatible with the
        # others.
        tshape = array(value.shape[1:4])
        vtype  = value.vtype
        if ( compat.checkNone(self.m_eshape) ):
            if ( vtype == "E" ):
                list.__setattr__(self,'m_eshape',tshape.copy())
                list.__setattr__(self,'m_nshape',tshape +1)
            else:
                list.__setattr__(self,'m_nshape',tshape.copy())
                list.__setattr__(self,'m_eshape',tshape -1)
        else:
            emtch = (tshape == self.m_eshape).all()
            nmtch = (tshape == self.m_nshape).all()
            if ( ( vtype == "E" ) and not ( emtch ) ):
                raise ValueError,\
                    "PIVVar has invalid eshape."
            elif ( ( vtype == "N" ) and not ( nmtch ) ):
                raise ValueError,\
                    "PIVVar has invalid nshape."

        # Store value.  Only cache if we are setting a PIVVar.
        if ( isinstance(value,PIVVar) ):
            ouuid = self.m_cache.store(value,self.__deactivate__)
            dict.__setitem__(self,key,ouuid)
        else:
            dict.__setitem__(self,key,value)

    def addVars(self,vars):
        """
        Adds a single PIVVar or list of PIVVars to PIVEpoch.
        """
        if ( not isinstance(vars,list) ):
            vars = [ vars ]

        for i in range(len(vars)):
            value = vars[i]       
            if ( not isinstance(value,PIVVar) and not isinstance(value,PVStore) ):
                raise ValueError,\
                    "PIVEpoch can only store PIVVar or PVStore objects."
            
            key = value.name
            self.__setitem__(key,value)
            
    def getCache(self):
        """
        Gets a reference to the cache object.  NOTE: This cache
        may be shared.
        """
        return self.m_cache

    def getVars(self):
        """
        Returns a list of all stored variables.
        """
        keys = self.keys()
        keys.sort()
        vars = []
        for i in range(len(keys)):
            vars.append( self.__getitem__(keys[i]) )

        return vars

    def printStats(self):
        """
        Prints summary statistics for all variables in PIVEpoch.
        """
        print "************ EPOCH %f [%s] : %i VARS ************" %\
            (self.m_time,self.m_units,len(self))
        keys = self.keys()
        keys.sort()
        print "|"
        for i in range(len(keys)):
            self.__getitem__(keys[i]).printStats()
            print "|"
           
    def rebuildKeys(self):
        """
        Force an update of PIVEpoch to ensure that keys match variable
        names.  rebuildKeys() will be automatically called if a key
        name isn't found in the dictionary.
        
        The primary purpose of this function is to accomodate variable
        renames (eg, when executing avar.setName() on a variable stored
        in the PIVEpoch).
        """
        keys = self.keys()
        for i in range(len(keys)):
            okey = keys[i]
            var  = self[okey]                
            if ( var.name != okey ):
                self.pop(okey)
                self.addVars(var)

    def setCache(self,cache):
        """
        Sets the cache object.
        """
        list.__setattr__(self,'m_cache',cache)
            
    def setTime(self,time,units=None):
        """
        Sets the floating point time value for the current epoch.
        """
        list.__setattr__(self,'m_time',float(time))
        if ( not compat.checkNone(units) ):
            self.setUnits(units)

    def setUnits(self,units):
        """
        Sets the units string for the time value.  units will be truncated
        to 31 characters for compatibility with ExodusII.
        """
        ul = min(len(units),self.m_max_str_len)

        list.__setattr__(self,'m_units',units[0:ul])       
        
         

#################################################################
#
class PIVData(list):
    """
    The PIVData container aggregates all PIV data and associated grid
    definition for a given experiment into one object for easy handling 
    and export.  A PIVData object, a subclass of Python's list, is simply
    a list of PIVEpochs.

    Data stored in a PIVData object can be accessed in several ways.  The
    most common are by using the syntax:
           aPIVData[x] ---------- Returns the PIVEpoch object at index x.
           aPIVData[x]['aVar'] -- Returns the variable named aVar for 
                                  Epoch x.

    PIVData objects come equipped with the ability to export their data
    in an ExodusII file format by calling the save() method.  The PIVData
    object can then be recreated later by calling the module function
    loadpivdata()
    
    The PIVData object provides a variable cache that is shared among
    the contained Epochs.  This shared cache greatly improves memory
    utilization efficiency.  The cache's size is determined by the 
    csize parameter.  Changing csize to a value other than the default
    should not be necessary.
    
    NOTE: Epoch time values should increase monotonically.  Exporting to
    ExodusII will not proceed otherwise.

    NOTE: Prior to exporting, the number of variables for each Epoch must 
    be the same, and variable names between Epoch's must be the same (ie, 
    if one Epoch has a variable named DISP, then all Epochs must have the 
    same variable). Exporting to any file format will not proceed otherwise.
    """
    def __init__(self,cellsz,origin,desc="PIVDATA",vars=None,exo=None,
                 csize=6,cache=None):
        """
        A PIVData object must be initialized with:
            cellsz --- /Float/ (z,y,x) cell size of the grid.
            origin --- /Float/ (z,y,x) coordinates for cell (0,0,0).
        
        Optionally the following can be supplied:
            desc ----- /String/ A description of the dataset.  desc
                       will be truncated to 79 characters.
            vars ----- /PIVVar or PIVEpoch/ A single PIVVar/PIVEpoch 
                       or a list of PIVVars/PIVEpochs.
            exo ------ /Object/ ExodusII file object.  The user does
                       not need to set exo as this is handled automatically.
            csize ---- /Int/ Cache size.  User should not need to modify
                       csize.
            cache ---- /PIVCache/ A PIVCache object for the Epoch to use.
                       If cache is provided, csize will be ignored.
                       
        If a single PIVVar object is passed, it will be stored for Epoch 0.
        If a 1D list of PIVVar's are passed, they will be stored for Epoch 0.
        Passing a 2D list of PIVVar's will store each row of PIVVar's for
        a different epoch.  For example 
            vars = [[epoch0_var0,epoch0_var1],[epoch1_var0,epoch1_var1]]
        will store the epoch0* variables for Epoch 0 and the epoch1*
        variables for Epoch 1.  Don't forget to set the epoch time values
        by calling setTimes().

        NOTE: All PIVVar objects of a given vtype must have the same
        shape.

        A single PIVEpoch or list of PIVEpochs can also passed during 
        initialization.
        """
        list.__setattr__(self,'m_max_str_len',ex2.EX_MAX_STR_LENGTH -1)
        list.__setattr__(self,'m_max_line_len',ex2.EX_MAX_LINE_LENGTH -1)

        self.setCellsz(cellsz)
        self.setOrigin(origin)
        self.setDesc(desc)
        list.__setattr__(self,'m_qa',[])
        
        if ( not compat.checkNone(cache) ):
            self.setCache(cache)
        else:
            self.setCache( PIVCache(csize) )

        # Append any vars if passed.
        if ( not compat.checkNone(vars) ):
            if ( isinstance(vars,list) ):
                if ( isinstance(vars[0],list) ):
                    for i in range( len(vars) ):
                        self.addVars(i,vars[i])
                elif ( isinstance(vars[0],PIVEpoch) ):
                    for i in range( len(vars) ):
                        self.append(vars[i])
                else:
                    self.addVars(0,vars)
            elif ( isinstance(vars,PIVEpoch) ):
                self.append(vars)
            else:
                self.addVars(0,vars)

        list.__setattr__(self,"m_exo",exo)

        # Add a QA record.
        date = datetime.datetime.now()
        qarec = ["SPIVET",
                 spivetrev.spivet_bld_rev,
                 date.date().isoformat(),
                 date.time().isoformat()]
        self.addQA(qarec)
        
    def __del__(self):
        # Can't close the EX2 file unless the refcount is less than 4
        # Since another Epoch in a different PIVData may have PVStore 
        # variables that depend on the exo object.  
        exo = self.m_exo
        if ( not compat.checkNone((exo)) and (sys.getrefcount(exo) < 4) ):
            ex2.ex_close(exo)
            
        # List has no __del__() method, so no need to call it.

    def __getattr__(self,name):
        if ( name == 'desc' ):
            return self.m_desc
        elif ( name == 'cellsz' ):
            return self.m_cellsz.copy()
        elif ( name == 'origin' ):
            return self.m_origin.copy()
        else:
            raise AttributeError, 'PIVData object has no attribute ' + name

    def __getitem__(self,key):
        if ( isinstance(key,slice) ):
            return PIVData( self.cellsz, 
                            self.origin, 
                            self.desc,
                            list.__getitem__(self,key) )
        else:
            return list.__getitem__(self,key)

    def __getslice__(self,i,j):
        """
        __getslice__() is deprecated and removed in Python 3.0, however
        for 2.x versions, any class that is derived from list will have
        its __getslice__() function called (if it exists) prior to 
        its __getitem__() function.  Therefore for PIVData to behave
        correctly, __getslice__() must call __getitem__().
        """
        return self.__getitem__(slice(i,j))
            
    def __setattr__(self,name,value):
        raise AttributeError,\
            "Please use methods to set PIVData attributes."
            
    def __timeismon__(self):
        """
        Checks that time is monotonic.
        """
        if ( len(self) <= 1 ):
            return True
        
        # Monotonically increasing.
        for i in range(1,len(self)):
            if ( self[i].time <= self[i-1].time ):
                return False

        return True

    def __varsvalid__(self):
        """
        Check that the number of variables and variable names are the 
        same among Epochs.
        """
        if ( len(self) <= 1 ):
            return True
        
        nvars  = len(self[0])
        varnms = self[0].keys()
        varnms.sort()
        for i in range(1,len(self)):
            if ( len(self[i]) != nvars ):
                return False
            vn = self[i].keys()
            vn.sort()
            for j in range(nvars):
                if ( vn[j] != varnms[j] ):
                    return False

        return True

    def addsQA(self,rec):
        """
        Adds a simplified QA record consisting of 1 user specified string, 
        rec, and 3 other system generated strings
           SPIVET Version
           Date
           Time

        addsQA() is meant to be a simplified interface to addQA().  For
        more details, see addQA().
        """
        date = datetime.datetime.now()
        frec = [rec, 
                "SPIVET R%i" % spivetrev.spivet_bld_rev, 
                date.date().isoformat(),
                date.time().isoformat()]

        self.addQA(frec)
        
    def addQA(self,rec):
        """
        PIVData objects provide a means to track creation and modification
        of the dataset by way of optional quality assurance records.  A 
        record is a list of 4 strings, each of which is a maximum of 31 
        characters in length.  The strings can contain any information, but 
        the following is the suggested format:
            rec[0] = Program name (eg, PivLIB)
            rec[1] = Program version (eg, 2.3)
            rec[2] = Date of modification
            rec[3] = Time of modifcation
        """
        if ( ( not isinstance(rec,list) ) or ( len(rec) != 4 ) ):
            raise ValueError,\
                "rec must be a list of 4 strings."
        
        for i in range(4):
            rec[i] = str(rec[i])
            rl     = min(len(rec[i]),self.m_max_str_len)
            rec[i] = rec[i][0:rl]

        self.m_qa.append(rec)

    def addVars(self,epndx,vars):
        """
        Adds variable to the Epoch at index epndx.  vars can be a 
        single PIVVar or a list of PIVVars.

        NOTE: If epndx is greater than the length of PIVData, a
        new Epoch will be added and the time of the Epoch will be
        set to epndx.
        """
        # Check length of self and add empty entries if necessry.
        if ( epndx >= len(self) ):
            for i in range(len(self),epndx+1):
                self.append(PIVEpoch(epndx,cache=self.m_cache))

        self[epndx].addVars(vars)            

    def append(self,epoch):
        """
        Add an epoch to the collection.
        """
        if ( not isinstance(epoch,PIVEpoch) ):
            raise ValueError %\
                "PIVData objects can only store PIVEpoch objects."

        ecache = epoch.getCache()
        if ( id(ecache) != id(self.m_cache) ):
            self.m_cache.assimilateCache(ecache)
            epoch.setCache(self.m_cache)

        list.append(self,epoch)

    def dump2ex2(self,ofpath,fprc=4):
        """
        Dumps data to an ExodusII data file.  ExodusII is the preferred 
        file format for exporting PIVData objects.

        fprc specifies the floating point precision in bytes for the
        output file.  Must be either 4 or 8.

        NOTE:  dump2ex2() will silently overwrite files at location ofpath.
        """
        # Initialization.
        if ( len(self) == 0 ):
            print "Nothing to export because PIVData is empty."
            return

        if ( ( fprc != 8 ) and ( fprc != 4 ) ):
            raise AttributeError,\
                "Floating point precision, fprc, must be 4 or 8."

        if ( not self.__timeismon__() ):
            raise ValueError,\
                "Epoch time values must increase monotonically."

        if ( not self.__varsvalid__() ):
            raise ValueError,\
                "Epochs must share a common number of variables, with each "\
                +"variable named consistently across Epochs."
        nvars  = len(self[0])
        varnms = self[0].keys()
        varnms.sort()

        eshape = self[0].eshape
        nshape = self[0].nshape

        origin = self.origin
        cellsz = self.cellsz

        ext = ofpath[(len(ofpath)-4):len(ofpath)]
        ext = ext.lower()
        if ( ext != ".ex2" ):
            ofpath = ofpath +".ex2"

        ntstps = len(self) 

        if ( fprc == 4 ):
            ifdtype = dtype(float32)
        else:
            ifdtype = dtype(float64)

        # Set up variable names and determine number of each variable vtype.
        vnvars    = [0,0]    # [elm,node]
        ex2nvars  = [0,0]
        vvarnms   = [[],[]]
        ex2varnms = [[],[]]
        vdtype    = []
        for i in range(nvars):
            var   = self[0][varnms[i]]
            vtype = var.vtype.upper()
            dim   = var.shape[0]

            vdtype.append(var.dtype)

            wvarnm = "%s [%s]" % (varnms[i],var.units)
            if ( vtype == "E" ):
                ndx = 0
            elif ( vtype == "N" ):
                ndx = 1
            else:
                continue

            vnvars[ndx] = vnvars[ndx] +1
            vvarnms[ndx].append(varnms[i]) 
            if ( dim == 1 ):
                ex2nvars[ndx] = ex2nvars[ndx] +1
                ex2varnms[ndx].append(wvarnm)
            elif ( dim == 3 ):
                ex2nvars[ndx] = ex2nvars[ndx] +3
                # ParaView's vector glomming currently requires vector
                # components to be written X, Y, Z.
                ex2varnms[ndx].append(wvarnm +" X")
                ex2varnms[ndx].append(wvarnm +" Y")
                ex2varnms[ndx].append(wvarnm +" Z")
            elif ( dim == 9 ):
                ex2nvars[ndx] = ex2nvars[ndx] +9
                ex2varnms[ndx].append(wvarnm +" XX")
                ex2varnms[ndx].append(wvarnm +" XY")
                ex2varnms[ndx].append(wvarnm +" XZ")
                ex2varnms[ndx].append(wvarnm +" YX")
                ex2varnms[ndx].append(wvarnm +" YY")
                ex2varnms[ndx].append(wvarnm +" YZ")
                ex2varnms[ndx].append(wvarnm +" ZX")
                ex2varnms[ndx].append(wvarnm +" ZY")
                ex2varnms[ndx].append(wvarnm +" ZZ")

        # Build mesh.
        elcon = pivlibc.bldex2msh(eshape)

        nzelms = eshape[0]
        nyelms = eshape[1]
        nxelms = eshape[2]

        nelms  = nzelms*nyelms*nxelms
        nnodes = nshape[0]*nshape[1]*nshape[2]

        nodcrdmat = indices(nshape,dtype=ifdtype)
        nodcrdz   = nodcrdmat[0].reshape(nnodes)
        nodcrdy   = nodcrdmat[1].reshape(nnodes)
        nodcrdx   = nodcrdmat[2].reshape(nnodes)

        nodcrdz = nodcrdz*cellsz[0] +origin[0] -cellsz[0]/2.
        nodcrdy = nodcrdy*cellsz[1] +origin[1] -cellsz[1]/2.
        nodcrdx = nodcrdx*cellsz[2] +origin[2] -cellsz[2]/2.

        # Need to write the data to a temporary EX2 file to avoid
        # overwriting a PVStore file with the same name.
        tfh = tempfile.mkstemp()
        os.close(tfh[0])

        # Write out data.  If the user specifies that the output file
        # should be single precision, then assume that the incoming
        # data will be single precision.  ExodusII wrapper will handle
        # conversion if assumption is wrong.
        exo = ex2.ex_create(tfh[1],ex2.exc['EX_CLOBBER'],fprc,fprc)

        ex2.ex_put_init(exo,self.desc,3,nnodes,nelms,1,0,0)

        qa = self.getQA()
        ex2.ex_put_qa(exo,len(qa),qa)

        infostr = ["PIVDATAOBJECT"]
        infostr.append( "TIME UNITS [%s]" % self[0].units )
        infostr.append( 
            "PIVDATA-XORIGIN:%s" % pickle.dumps( float(origin[2]) ) )
        infostr.append( 
            "PIVDATA-YORIGIN:%s" % pickle.dumps( float(origin[1]) ) )
        infostr.append( 
            "PIVDATA-ZORIGIN:%s" % pickle.dumps( float(origin[0]) ) )
        infostr.append( 
            "PIVDATA-XCELLSZ:%s" % pickle.dumps( float(cellsz[2]) ) )
        infostr.append( 
            "PIVDATA-YCELLSZ:%s" % pickle.dumps( float(cellsz[1]) ) )
        infostr.append( 
            "PIVDATA-ZCELLSZ:%s" % pickle.dumps( float(cellsz[0]) ) )        
        for i in range( nvars ):
            infostr.append("%s.DTYPE:%s" % (varnms[i],vdtype[i].name) )

        ex2.ex_put_info(exo,len(infostr),infostr)

        ex2.ex_put_coord(exo,nodcrdx,nodcrdy,nodcrdz)
        ex2.ex_put_coord_names(exo,["X","Y","Z"])

        ex2.ex_put_elem_block(exo,1,"HEX",nelms,8,0)
        ex2.ex_put_elem_conn(exo,1,elcon)

        if ( ex2nvars[0] > 0 ):
            ex2.ex_put_var_param(exo,"E",ex2nvars[0])
            ex2.ex_put_var_names(exo,"E",ex2nvars[0],ex2varnms[0])
        if ( ex2nvars[1] > 0 ):
            ex2.ex_put_var_param(exo,"N",ex2nvars[1])
            ex2.ex_put_var_names(exo,"N",ex2nvars[1],ex2varnms[1])

        # ParaView animates based on the timestamp.  This creates an
        # incompatibility between files that support a timestamp and those
        # that don't.  To work around, the floating point time stamp will 
        # be stored as a global variable, and ex_put_time() will be called
        # with an integer value.
        times = self.getTimes()
        ex2.ex_put_var_param(exo,"G",1)
        ex2.ex_put_var_names(exo,"G",1,["EPOCHTIME"])
        for i in range(ntstps):
            ex2.ex_put_time(exo,i+1,i)
            ex2.ex_put_glob_vars(exo,i+1,1,[times[i]])

        if ( ex2nvars[0] > 0 ):
            ex2.ex_put_elem_var_tab(exo,1,ex2nvars[0],
                                    ones(ex2nvars[0],dtype=int))

        ex2.ex_put_prop_names(exo,ex2.exc["EX_ELEM_BLOCK"],3,
                              ["NXCELLS","NYCELLS","NZCELLS"])
        ex2.ex_put_prop(exo,ex2.exc["EX_ELEM_BLOCK"],1,"NXCELLS",nxelms)
        ex2.ex_put_prop(exo,ex2.exc["EX_ELEM_BLOCK"],1,"NYCELLS",nyelms)
        ex2.ex_put_prop(exo,ex2.exc["EX_ELEM_BLOCK"],1,"NZCELLS",nzelms)

        # Element data.
        for e in range(ntstps):
            ex2ndx = 1
            for v in range(vnvars[0]):
                vn   = vvarnms[0][v] 
                var  = self[e][vn]
                ncmp = var.shape[0]
                # Flip vtype components to x, y, z order.
                if ( ncmp == 1 ):
                    oset = 0
                else:
                    oset = ncmp -1
                for d in range(ncmp):
                    data = var[oset-d,:,:,:]
                    ex2.ex_put_elem_var(exo,e+1,ex2ndx,1,nelms,
                                        data.reshape(data.size))
                    ex2ndx = ex2ndx +1

        # Nodal data.
        for e in range(ntstps):
            ex2ndx = 1
            for v in range(vnvars[1]):
                vn   = vvarnms[1][v] 
                var  = self[e][vn]
                ncmp = var.shape[0]
                # Flip vtype components to x, y, z order.
                if ( ncmp == 1 ):
                    oset = 0
                else:
                    oset = ncmp -1
                for d in range(ncmp):
                    data = var[oset-d,:,:,:]
                    ex2.ex_put_nodal_var(exo,e+1,ex2ndx,nnodes,
                                         data.reshape(data.size))
                    ex2ndx = ex2ndx +1

        ex2.ex_close(exo)
        shutil.move(tfh[1], ofpath)
        os.chmod(ofpath,stat.S_IRUSR|stat.S_IWUSR|stat.S_IRGRP|stat.S_IROTH)

    def extend(self, epochs):
        """
        Appends a list of Epochs to the collection.
        """
        if ( isinstance(epochs,list) ):
            for i in range( len(epochs) ):
                self.append(epochs[i])
        else:
            self.append(epochs)
                        
    def getCache(self):
        """
        Gets a reference to the cache object.  NOTE: This cache
        may be shared.
        """
        return self.m_cache            

    def getQA(self):
        """
        Returns a list of QA records.
        """
        return self.m_qa

    def getTimes(self):
        """
        Returns a list of the epoch times.
        """
        times = []
        for i in range(len(self)):
            times.append(self[i].time)

        return times
    
    def save(self, ofpath, fprc=4):
        """
        Saves a copy of the PIVData object for analysis with ParaView
        or later work with SPIVET.  Calling save() is identical to calling
        dump2ex2().

        fprc specifies the floating point precision in bytes to use when 
        writing the file.  fprc must be either 4 or 8.
        """
        self.dump2ex2(ofpath,fprc)
        
    def setCache(self,cache):
        """
        Sets the cache object.
        """
        list.__setattr__(self,'m_cache',cache)
        
        for e in self:
            ecache = e.getCache()
            if ( id(ecache) != id(cache) ):
                cache.assimilateCache(ecache)
                e.setCache(cache)

    def setCellsz(self, cellsz):
        """
        Sets the grid (z,y,x) cell size.
        """
        cellsz = array(cellsz,dtype=float)
        if ( cellsz.size != 3 ):
            raise ValueError,\
                "cellsz must be a 3-element tuple of (z,y,x) cell sizes."

        list.__setattr__(self,"m_cellsz",cellsz)

    def setDesc(self, desc):
        """
        Sets the PIVData description string.  desc will be truncated to
        79 characters.
        """
        if ( not isinstance(desc,str) ):
            raise ValueError,\
                "desc must be a string."
        dl = min(len(desc),self.m_max_line_len)

        list.__setattr__(self,"m_desc",desc[0:dl])

    def setOrigin(self, origin):
        """
        Sets the grid (z,y,x) origin for cell (0,0,0).
        """
        origin = array(origin,dtype=float)
        if ( origin.size != 3 ):
            raise ValueError,\
                "origin must be a 3-element tuple with origin of cell (0,0,0)."

        list.__setattr__(self,"m_origin",origin)

    def setTime(self, epndx, time):
        """
        Sets the time for the Epoch at epndx to time.
        """
        if ( epndx >= len(self) ):
            raise ValueError,\
                "Epoch %i does not exist." % epndx

        self[epndx].setTime(time)

    def setTimes(self, times):
        """
        Sets the times for all epochs to the list of values in times.
        """
        times = array(times)
        if ( times.size != len(self) ):
            raise ValueError,\
                "The number of time values must match the number of epochs."

        for i in range( len(self) ):
            self[i].setTime(times[i])

    def setTimeUnits(self, units):
        """
        Sets the time units.
        """
        if ( len(self) == 0 ):
            raise ValueError,\
                "PIVData is empty and cannot set time units."

        for i in range( len(self) ):
            self[i].setUnits(units)


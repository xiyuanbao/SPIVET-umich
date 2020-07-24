"""
Filename:  floutil.py
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
  Module containing utility routines for FloLIB.

Contents:
  d_di()
  rk2()
  rk4()
"""

from spivet.pivlib import pivdata

from numpy import *

#################################################################
#
def d_di(var,axis,icellsz):
    """
    ----
    
    var            # Variable to be differentiated.
    axis           # Axis along which to differentiate.
    icellsz        # Cell size of grid along axis.
    
    ----
    
    Computes the spatial first derivative of var along the
    specified axis using central central differences for interior 
    cells.  The derivatives at the edge cells are computed using 
    forward or backward differences.  

    var is expected to have a shape of either [ncmp,nz,ny,nx]
    (ie, the shape of a PIVVar) or [nz,ny,nx], where ncmp is the
    number of vector components, and nz,ny,nz are the number of
    data points in z,y,x.  Note: ncmp can be either 1 or 3.

    On interior cells, the derivative is given by:
        d/di[i] = ( var[i+1] - var[i-1] )/(2 * icellsz)
    
    On edge cells, the derivative is given by:
        d/di[i] = ( var[i+1] - var[i] )/icellsz, or
        d/di[i] = ( var[i] -var[i-1] )/icellsz

    axis specifies which spatial axis the derivatives will be
    computed over.  axis can be one of:
        0 --- z-axis (ie, compute d/dz)
        1 --- y-axis (ie, compute d/dy)
        2 --- x-axis (ie, compute d/dx)

    Returns a PIVVar, drvtv, containing the specified derivative.
    """
    # Initialization.
    if ( var.ndim == 3 ):
        ncmp   = 1
        vshape = array([1,var.shape[0],var.shape[1],var.shape[2]])
    elif ( var.ndim == 4 ):
        ncmp   = var.shape[0]
        vshape = var.shape
    else:
        raise ValueError("Dimension of var must be 3 or 4.")

    vptr = var
    vptr = vptr.reshape(vshape)

    if ( ncmp > 3 ):
        raise ValueError("d_di does not support PIVVar's with ncmp > 3.")

    nz = vshape[1]
    ny = vshape[2]
    nx = vshape[3]

    drvtv = pivdata.PIVVar(vshape)
    if ( isinstance(var,pivdata.PIVVar) ):
        if ( axis == 0 ):
            dis = "Z"
        elif ( axis == 1 ):
            dis = "Y"
        else:
            dis = "X"

        name  = "D(%s)_D%s" % (var.name,dis)
        units = "(%s)_MM" % var.units

        drvtv.setAttr(name,units,var.vtype)

    # Compute the derivative.
    fosf = 1./icellsz
    sosf = 0.5*fosf
    if ( axis == 0 ):  # d/dz
        # Edge cells.
        drvtv[:,0,:,:]    = fosf*( vptr[:,1,:,:] -vptr[:,0,:,:] )
        drvtv[:,nz-1,:,:] = fosf*( vptr[:,nz-1,:,:] -vptr[:,nz-2,:,:] )
        
        # Inner cells.
        drvtv[:,1:(nz-1),:,:] = sosf*( vptr[:,2:nz,:,:] 
                                       -vptr[:,0:(nz-2),:,:] )

    elif ( axis == 1):  # d/dy
        # Edge cells.
        drvtv[:,:,0,:]    = fosf*( vptr[:,:,1,:] -vptr[:,:,0,:] )
        drvtv[:,:,ny-1,:] = fosf*( vptr[:,:,ny-1,:] -vptr[:,:,ny-2,:] )

        # Inner cells.
        drvtv[:,:,1:(ny-1),:] = sosf*( vptr[:,:,2:ny,:]
                                       -vptr[:,:,0:(ny-2),:] )

            
    else:               # d/dx
        # Edge cells.
        drvtv[:,:,:,0]    = fosf*( vptr[:,:,:,1] -vptr[:,:,:,0] )
        drvtv[:,:,:,nx-1] = fosf*( vptr[:,:,:,nx-1] -vptr[:,:,:,nx-2] )
        
        # Inner cells.
        drvtv[:,:,:,1:(nx-1)] = sosf*( vptr[:,:,:,2:nx]
                                       -vptr[:,:,:,0:(nx-2)] )


    return drvtv


#################################################################
#
def rk2(var,dt,fun,fargs=None):
    """
    ----
    
    var             # Variable to be incremented.
    dt              # Step increment.
    fun             # RHS of dvar/dt.
    fargs=None      # List of arguments to be passed to fun.
    
    ----
        
    Performs a second-order Runge-Kutta time step on the equation
        dvar/dt = fun(var,st,fargs)

    The RK2 method increments the above equation in a two-step process:
        ivar = var + (dt/2)*fun(var,0,fargs)
        ivar = var + (dt)*fun(ivar,dt/2,fargs)

    Note: If fun is an explicit function of time, then t at the start
    of the step should be passed in fargs.  fun can then compute the
    intermediate time as, t* = t +st.

    var can be any 1D numpy array.

    fun can be any Python function with the following prototype
        fun(var,st,fargs)
    where st is the incremental step time and fargs is a list of 
    additional arguments to be passed to fun.
    
    Returns ivar.
    """
    st = dt/2.
    ivar = var +st*fun(var,0,fargs)
    ivar = var +dt*fun(ivar,st,fargs)
    
    return ivar


#################################################################
#
def rk4(var,dt,fun,fargs=None):
    """
    ----
    
    var,            # Variable to be incremented.
    dt,             # Step increment.
    fun,            # RHS of dvar/dt.
    fargs=None      # List of arguments to be passed to fun.

    ----
    
    Performs a fourth-order Runge-Kutta time step on the equation
        dvar/dt = fun(var,st,fargs)

    The RK4 method increments the above equation in a four-step process:
        ivara = var + (dt/2)*fun(var,0,fargs)
        ivarb = var + (dt/2)*fun(ivara,dt/2,fargs)
        ivarc = var + dt*fun(ivarb,dt/2,fargs)
        ivar  = var + (dt/6)*( fun(var,0,fargs) +2*fun(ivara,dt/2,fargs)
                              +2*fun(ivarb,dt/2,fargs) +fun(ivarc,dt,fargs) )

    Note: If fun is an explicit function of time, then t at the start
    of the step should be passed in fargs.  fun can then compute the
    intermediate time as, t* = t +st.

    var can be any 1D numpy array.

    fun can be any Python function with the following prototype
        fun(var,st,fargs)
    where st is the incremental step time and fargs is a list of 
    additional arguments to be passed to fun.
    
    Returns ivar.
    """
    std2 = dt/2.
    std6 = dt/6.
    
    fun0  = fun(var,0,fargs)
    ivara = var +std2*fun0
    
    fun1  = fun(ivara,std2,fargs)
    ivarb = var +std2*fun1
    
    fun2  = fun(ivarb,std2,fargs)
    ivarc = var +dt*fun2
    
    fun3 = fun(ivarc,dt,fargs)
    ivar = var +std6*( fun0 +2.*fun1 +2.*fun2 +fun3)
    
    return ivar

    
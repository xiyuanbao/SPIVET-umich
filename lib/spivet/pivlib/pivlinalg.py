"""
Filename:  pivlinalg.py
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
  Module containing streamlined LAPACK wrappers.  The wrappers 
  defined here are very similar to those of SciPy, but without 
  some of the setup and error-checking overhead (which can be 
  very time consuming).

Contents:
  dsolve()
  fsolve()
  getdgesv()
  solve()
"""

from numpy import *
from scipy import linalg
from scipy.linalg import lapack


#################################################################
#
def getdgesv():
    dm = zeros((2,2),dtype=float)

    [dgesv] = lapack.get_lapack_funcs(['gesv'],(dm,dm))

    return dgesv


#################################################################
#
__dgesv = getdgesv()


#################################################################
#
def dsolve(amat,b):
    """
    ----
    
    amat           # n x n system matrix.
    b              # n x l array of right-hand sides.

    ----

    Solves amat*x = b using SciPy's wrapper for LAPACK's dgesv. 

    b can be an n-element array, or a matrix of l right-hand sides
    (each of length n).

    Returns an n x l array of solution vectors. 
    """
    [lu, p, x, info] = __dgesv(amat,b,0,0) 

    if ( info > 0 ):
        raise linalg.LinAlgError

    return x


#################################################################
#
def fsolve(amat,b,fcn):
    """
    ----

    amat           # n x n system matrix.
    b              # n x l array of right-hand sides.
    fcn            # SciPy LAPACK function.
    
    ----
    
    Solves amat*x = b.

    b can be an n-element array, or a matrix of l right-hand sides
    (each of length n).

    fcn is a function determined by calling
        [fcn] = scipy.linalg.lapack.get_lapack_funcs(['gesv'],(amat,b))
    Specifying fcn directly can save significant computation time
    if fsolve is called repetively.
        
    Returns an n x l array of solution vectors. 
    """

    [lu, p, x, info] = fcn(amat,b,0,0) 

    if ( info > 0 ):
        raise linalg.LinAlgError

    return x


#################################################################
#
def solve(amat,b):
    """
    ----

    amat           # n x n system matrix.
    b              # n x l array of right-hand sides.
    
    ----
        
    Solves amat*x = b. 

    b can be an n-element array, or a matrix of l right-hand sides
    (each of length n).

    Returns an n x l array of solution vectors. 
    """
    [gesv] = lapack.get_lapack_funcs(['gesv'],(amat,b))
    
    [lu, p, x, info] = gesv(amat,b,0,0) 

    if ( info > 0 ):
        raise linalg.LinAlgError

    return x



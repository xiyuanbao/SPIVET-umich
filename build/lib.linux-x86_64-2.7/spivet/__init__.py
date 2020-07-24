"""
Filename:  spivet.py
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
  The SPIVET package provides a set of libraries for the
  analysis of flows using stereoscopic particle image velocimetry
  combined with thermochromic thermometry.  

Contents:
  steps ------ The steps module provides the preferred means of
               utilizing the SPIVET library.  The user can
               construct an analytical recipe from a set of 
               discrete steps.  See documentation on the steps
               module for more details.

  pivlib ----- Provides the underpinnings for (stereoscopic)
               particle image velocimetry.  Also provides the 
               fundamental data classes used for SPIVET.
  tlclib ----- Provides thermochromic thermometry functionality.
  flolib ----- High level post processing library.  Takes results
               from pivlib and tlclib, fitlers the data, and 
               constructs other flow field variables.
"""

import pivlib

import tlclib

import flolib

import spivetrev
from spivetrev import spivet_bld_rev

import steps
import compat

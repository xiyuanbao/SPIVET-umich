"""
Filename:  __init__.py
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
  TlcLIB is a library for processing temperature data gained
  from the use of thermochromic liquid crystals. 

  TlcLIB is meant to be used in cooperation with PivLIB, and as
  such makes use of PivLIB data structures (eg, pivdict).

  At present, TlcLIB only utilizes a single camera for
  thermochromic calibration and temperature field extraction.
"""

import tlctccal
from tlctccal import calibrate, loadtcpa, loadtlccal, savetlccal

import tlctc
from tlctc import hue2tmp

import tlctf
from tlctf import tfcomp

import tlcutil
from tlcutil import tlcmask

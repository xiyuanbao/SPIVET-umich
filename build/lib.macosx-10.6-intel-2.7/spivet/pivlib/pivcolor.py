"""
Filename:  pivcolor.py
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
  Module containing colormaps for use with Pylab.

  The colormaps can be accessed via the pivlib.cm namespace.
  
Contents:
  pivmap ---- Equivalent to getpivcmap()
  mpivap ---- Equivalent to getpivcmap(1)
  whsv ------ Similar to pylab's hsv map, except that the highest z-value
              gets mapped to white.
  wcontrast - High contrast colormap with highest value set to white.
"""

import pivutil
from matplotlib import colors

#################################################################
#
def _bld_whsv():
    """
    Builds the whsv colormap.  The values are taken from pylab's
    default hsv map.
    """
    red = [ (0.0, 1.0, 1.0), 
            (0.15873000000000001, 1.0, 1.0), 
            (0.17460300000000001, 0.96875, 0.96875), 
            (0.33333299999999999, 0.03125, 0.03125), 
            (0.34920600000000002, 0.0, 0.0), 
            (0.66666700000000001, 0.0, 0.0), 
            (0.68254000000000004, 0.03125, 0.03125),
            (0.84126999999999996, 0.96875, 0.96875), 
            (0.85714299999999999, 1.0, 1.0), 
            (0.99, 1.0, 1.0), 
            (1.0, 1.0, 1.0)]
    
    green = [ (0.0, 0.0, 0.0), 
              (0.15873000000000001, 0.9375, 0.9375), 
              (0.17460300000000001, 1.0, 1.0), 
              (0.50793699999999997, 1.0, 1.0), 
              (0.66666700000000001, 0.0625, 0.0625), 
              (0.68254000000000004, 0.0, 0.0), 
              (0.99, 0.0, 0.0), 
              (1.0, 1.0, 1.0)]
    
    blue = [ (0.0, 0.0, 0.0), 
             (0.33333299999999999, 0.0, 0.0), 
             (0.34920600000000002, 0.0625, 0.0625), 
             (0.50793699999999997, 1.0, 1.0), 
             (0.84126999999999996, 1.0, 1.0), 
             (0.85714299999999999, 0.9375, 0.9375), 
             (0.99, 0.09375, 0.09375), 
             (1.0, 1.0, 1.0) ]
    
    cdata = {'red':red,'green':green,'blue':blue}
    
    return colors.LinearSegmentedColormap('whsv',cdata,256)


#################################################################
#
def _bld_wcontrast():
    """
    Builds the wcontrast colormap.  
    """
    red = [ (0,0,0),
            (.125,0,0),
            (0.25,0.97,0.97),
            (0.375,1.,1.),
            (0.5,0,0),
            (0.625,1.,1.),
            (0.75,0,0),
            (0.875,1.,1.),
            (0.99,0,0),
            (1,1,1) ]

    green = [ (0,0,0),
              (0.125,1,1),
              (0.25,0,0),
              (0.375,1,1),
              (0.5,0,0),
              (0.625,0,0),
              (0.75,0.33,0.33),
              (0.875,0.54,0.54),
              (0.99,0.78,0.78),
              (1,1,1) ]
     
    blue = [ (0,0,0),
             (0.125,0,0),
             (0.25,0.78,0.78),
             (0.375,0.09,0.09),
             (0.5,1,1),
             (0.625,0,0),
             (0.75,0.09,0.09),
             (0.875,0,0),
             (0.99,1.,1.),
             (1,1,1) ]

    cdata = {'red':red,'green':green,'blue':blue}
    
    return colors.LinearSegmentedColormap('wcontrast',cdata,256)
    
#################################################################
#
pivmap    = pivutil.getpivcmap()
mpivmap   = pivutil.getpivcmap(1)
whsv      = _bld_whsv()
wcontrast = _bld_wcontrast()
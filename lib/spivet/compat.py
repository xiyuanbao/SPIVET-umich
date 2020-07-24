"""
Filename:  compat.py
Copyright (C) 2020 Xiyuan Bao
 
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
    Compatibility fix for python 2.7.
"""
# return True if a list or ndarray has None item, or x is None.
# Input could be int/float,list,string, ndarray and None. 
def checkNone(x):
    try:
        result = not all(x)
    except:
        result= x is None
    return result

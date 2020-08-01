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
import numpy as np

def checkNone(x):
    """
    return True if a list or ndarray has None item, or x is None, or np.nan!.
    Input could be int/float,list,string, ndarray, a function, and None.
    """
    if x is None:
        return True

    else:
        try:

                listlike = x[0]
                #result = not all(x) #does not work because of 0
                for item in x:
                        if item is None:
                                return True

                try:
                        minx = np.min(x)
                        result = np.isnan(minx) or np.isinf(minx)
                        return result
                except:

                        return result
        except:
                try:

                        result = np.isnan(x) or np.isinf(x)
                        return result
                except:
                        return False



""" 
def checkNone(x):
    try:
        result1 = not all(x)# True if None in a list or ndarray
	try:
		result2 = np.isnan(np.min(x))# True if np.nan in a list or ndarray
	except:
		result2= False #with flexible type like String, do not check nan
	result = result1 or result2
    except:
        result= x is None or x is np.nan
    return result
"""

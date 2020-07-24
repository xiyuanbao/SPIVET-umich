"""
Filename:  exodusII.py
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
    Provides exodusII data file format functionality for PivLIB.
"""

from ex2lib import *

# Dictionary of constants for API.
exc = {
    'EX_NOCLOBBER':0,
    'EX_CLOBBER':1,
    'EX_NORMAL_MODEL':2,
    'EX_LARGE_MODEL':4,
    'EX_NETCDF4':8,
    'EX_NOSHARE':16,
    'EX_SHARE':32,
    'EX_READ':0,
    'EX_WRITE':1,
    'EX_INQ_FILE_TYPE':1,
    'EX_INQ_API_VERS':2,
    'EX_INQ_DB_VERS':3,
    'EX_INQ_TITLE':4,
    'EX_INQ_DIM':5,
    'EX_INQ_NODES':6,
    'EX_INQ_ELEM':7,
    'EX_INQ_ELEM_BLK':8,
    'EX_INQ_NODE_SETS':9,
    'EX_INQ_NS_NODE_LEN':10,
    'EX_INQ_SIDE_SETS':11,
    'EX_INQ_SS_NODE_LEN':12,
    'EX_INQ_SS_ELEM_LEN':13,
    'EX_INQ_QA':14,
    'EX_INQ_INFO':15,
    'EX_INQ_TIME':16,
    'EX_INQ_EB_PROP':17,
    'EX_INQ_NS_PROP':18,
    'EX_INQ_SS_PROP':19,
    'EX_INQ_NS_DF_LEN':20,
    'EX_INQ_SS_DF_LEN':21,
    'EX_INQ_LIB_VERS':22,
    'EX_INQ_EM_PROP':23,
    'EX_INQ_NM_PROP':24,
    'EX_INQ_ELEM_MAP':25,
    'EX_INQ_NODE_MAP':26,
    'EX_ELEM_BLOCK':1,
    'EX_NODE_SET':2,
    'EX_SIDE_SET':3,
    'EX_ELEM_MAP':4,
    'EX_NODE_MAP':5
}

# Miscellaneous constants.
EX_MAX_STR_LENGTH  = 32
EX_MAX_LINE_LENGTH = 80

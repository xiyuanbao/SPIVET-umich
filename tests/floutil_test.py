"""
Filename:  floutil_test.py
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
    Runs various validation tests on the floutil module.
"""

from spivet import pivlib, flolib
from numpy import *
import hashlib, os, unittest, StringIO, sys
from os import path

class test_floutil(unittest.TestCase):
    def setUp(self):
        self.eps = 1.e-6
        
        self.dfp = "data"  # Input data file path.

        self.ofp = "test-output"  # Output file path
        if ( not path.exists(self.ofp) ):
            os.mkdir(self.ofp)

        # Redirect stdout for chatty functions.
        self.sobfr = StringIO.StringIO()
        sys.stdout = self.sobfr        

    def tearDown(self):
        self.sobfr.close()
        sys.stdout = sys.__stdout__       

    def test_d_di(self):
        a = array([[ [1,2,3,4],
                     [5,6,7,8],
                     [9,10,11,12],
                     [13,14,15,16] ],
        
                   [ [17,18,19,20],
                     [21,22,23,24],
                     [25,26,27,28],
                     [29,30,31,32] ],
        
                   [ [33,34,35,36],
                     [37,38,39,40],
                     [41,42,43,44],
                     [45,46,47,48] ],
        
                   [ [49,50,51,52],
                     [53,54,55,56],
                     [57,58,59,60],
                     [61,62,63,64] ]], dtype='float' )
        
        sf = 2.2
        aa = a*a
        
        dadz = flolib.d_di(a,0,sf)
        dady = flolib.d_di(a,1,sf)
        dadx = flolib.d_di(a,2,sf)
        
        daadz = flolib.d_di(aa,0,sf)
        daady = flolib.d_di(aa,1,sf)
        daadx = flolib.d_di(aa,2,sf)
        
        ddi  = array([[dadz,dady,dadx],[daadz,daady,daadx]])
        
        kddi = pivlib.pklload("%s/floutil-d_di-known" % self.dfp)
        
        d = abs(ddi -kddi)
        d = d < self.eps

        self.assertTrue( d.all() )
        
def suite():
    suite = unittest.TestSuite()
    suite.addTest( unittest.makeSuite( test_floutil ) )
    
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(suite())        



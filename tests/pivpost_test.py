"""
Filename:  pivpost_test.py
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
    Runs various validation tests on the pivpost module.
"""

from spivet import pivlib
from numpy import *
import os, unittest, StringIO, sys
from os import path


class test_medfltr(unittest.TestCase):
    def setUp(self):
        self.eps = 1.e-6
        
        self.dfp = "data"  # Input data file path.

        self.ofp = "test-output"  # Output file path
        if ( not path.exists(self.ofp) ):
            os.mkdir(self.ofp)

        # Redirect stdout for chatty functions.
        self.sobfr = StringIO.StringIO()
        sys.stdout = self.sobfr        

        self.kpd = pivlib.loadpivdata("%s/PIVDATA-known.ex2" % self.dfp)

    def tearDown(self):
        self.sobfr.close()
        sys.stdout = sys.__stdout__       

    def testPlanarTrue(self):
        [fof,fltrd] = pivlib.medfltr(self.kpd[0]["OFDISP3D"],5,3.,0.,True)
        d = abs(fof -self.kpd[0]["OFDISP3D-MEDFLTRD-PLNR"])
        d = d < self.eps
        self.assertTrue( d.all() )
        
        d = abs(fltrd -self.kpd[0]["MEDFLTRDFLAG-PLNR"])
        d = d < self.eps
        self.assertTrue( d.all() )

    def testPlanarFalse(self):
        [fof,fltrd] = pivlib.medfltr(self.kpd[0]["OFDISP3D"],5,3.,0.,False)
        d = abs(fof -self.kpd[0]["OFDISP3D-MEDFLTRD"])
        d = d < self.eps
        self.assertTrue( d.all() )
        
        d = abs(fltrd -self.kpd[0]["MEDFLTRDFLAG"])
        d = d < self.eps
        self.assertTrue( d.all() )

def suite():
    suite = unittest.TestSuite()
    suite.addTest( unittest.makeSuite( test_medfltr ) )
    
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(suite())


"""
Filename:  pivdata_test.py
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
    Runs various validation tests on the pivpgcal and pivpg 
    modules.
"""

from spivet import pivlib
from numpy import *
import hashlib, os, unittest, StringIO, sys
from os import path

class test_pivpg(unittest.TestCase):
    def setUp(self):
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

    def testCalibration(self):
        # Because camcal is determined by a non-linear optimization 
        # algorithm, different architectures can produce calibration 
        # parameters that differ by ~ 1% or less.  
        eps = 1.e-2
        
        cutoff = 1.e-2
                
        kcamcal = pivlib.loadcamcal("%s/CAMCAL_CAM1-known" % self.dfp)

        ccpa           = pivlib.loadccpa("%s/CC_OUT_CAM1-known" % self.dfp)
        ncamcal        = pivlib.initcamcal(4.65e-3,4.65e-3)
        [ocamcal,oerr] = pivlib.calibrate(ccpa,ncamcal,1000)

        keys = kcamcal.keys()
        for k in keys:
            kkv = kcamcal[k]
            okv = ocamcal[k]
        
            pt = 10.**floor(log10( abs(kkv) +1.E-16 ))
        
            nkkv = kkv/pt
            nokv = okv/pt
        
            d = abs(nokv -nkkv)
            d = array( d < eps )
            
            # Skip anything smaller than cutoff as these parameters tend
            # to have variation of up to 30% from one platform to the next.
            try:
                lpt = len(pt)
                msk = pt > cutoff
                self.assertTrue( d[msk].all(), "CAMCAL %s" % k )

            except TypeError:
                if ( pt < cutoff ):
                    continue

                self.assertTrue( d.all(), "CAMCAL %s" % k )            
                    
    def testWICSP(self):
        eps    = 1.e-4        
        cutoff = 1.e-5
                
        kcamcal = pivlib.loadcamcal("%s/CAMCAL_CAM1-known" % self.dfp)
        kwicsp  = pivlib.loadwicsp("%s/WICSP_CAM1-known" % self.dfp)
        rbndx   = [ [0,768], [0,1024] ]
        
        wdsc  = pivlib.dscwrld(rbndx,kcamcal) 
        wicsp = pivlib.wrld2imcsp(rbndx,kcamcal,wdsc)
        
        keys = kwicsp.keys()
        for k in keys:
            eps    = 1.E-3
            cutoff = 1.E-4
            
            kkv = kwicsp[k]
            okv = wicsp[k]

            pt = 10.**floor(log10( abs(kkv) +1.E-16 ))
            
            nkkv = kkv/pt
            nokv = okv/pt

            d = abs(nokv - nkkv)
            d = array( d < eps )

            try:
                lpt = len(pt)
                msk = pt > cutoff
                self.assertTrue( d[msk].all(), "WICSP %s" % k )
            except:
                self.assertTrue( d.all(), "WICSP %s" % k )
                

def suite():
    suite = unittest.TestSuite()
    suite.addTest( unittest.makeSuite( test_pivpg ) )
    
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(suite())


    

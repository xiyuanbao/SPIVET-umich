"""
Filename:  flotrace_test.py
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
    Runs various validation tests on the flotrace module.
"""

from numpy import *
from spivet import flolib, pivlib

import unittest, StringIO, sys

class test_floftle(unittest.TestCase):
    def setUp(self):
        self.eps = 1.e-6

        # Redirect stdout for chatty functions.
        self.sobfr = StringIO.StringIO()
        sys.stdout = self.sobfr

        self.velkey   = "VEL" 
        self.tssdiv   = 12
        self.eisz     = 15
        self.ntrpc    = [1,4,4]
        self.ncalls   = 1
        self.irsbpath = "test-output"

        pdifname = "data/TGYRE.ex2"        
        self.pd  = pivlib.loadpivdata(pdifname)

    def tearDown(self):
        self.sobfr.close()
        sys.stdout = sys.__stdout__

    def testForwardAdvection(self):
        irspath = "%s/%s" % (self.irsbpath,"FTLE-FWD")
        epslc   = slice(0,1,1)
        kpd     = pivlib.loadpivdata("data/TFTLE-FWD.ex2")
        
        ftledict = flolib.ftleinit(self.pd,
                                   epslc,
                                   self.eisz,
                                   self.ntrpc,
                                   self.ncalls,
                                   self.tssdiv)
        flolib.ftletrace(self.pd,self.velkey,irspath,ftledict)
        fpd = flolib.ftlecomp(irspath,ftledict)
        
        kvar = kpd[0]['FTLE']
        fvar = fpd[0]['FTLE']
        
        fdlta = abs(fvar -kvar)
        self.assertEqual( (fdlta < self.eps).all(), True )

    def testReverseAdvection(self):
        irspath = "%s/%s" % (self.irsbpath,"FTLE-BWD")
        epslc   = slice(15,14,-1)
        kpd     = pivlib.loadpivdata("data/TFTLE-BWD.ex2")
        
        ftledict = flolib.ftleinit(self.pd,
                                   epslc,
                                   self.eisz,
                                   self.ntrpc,
                                   self.ncalls,
                                   self.tssdiv)
        flolib.ftletrace(self.pd,self.velkey,irspath,ftledict)
        fpd = flolib.ftlecomp(irspath,ftledict)
        
        kvar = kpd[0]['FTLE']
        fvar = fpd[0]['FTLE']
        
        rdlta = abs(fvar -kvar)
        self.assertEqual( (rdlta < self.eps).all(), True)

def suite():
    suite = unittest.TestSuite()
    suite.addTest( unittest.makeSuite( test_floftle ) )
    
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(suite())


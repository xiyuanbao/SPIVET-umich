"""
Filename:  run_tests.py
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
    High level driver to run regression tests.  Execute
        python run_tests.py
    to run all tests.
"""

import unittest

import pivdata_test
import pivir_test
import pivpg_test
import pivpost_test
import pivsim_test
import floutil_test
import flotrace_test
import tlclib_test

def suite():
    suite = unittest.TestSuite()
    suite.addTest( pivdata_test.suite()  )
    suite.addTest( pivir_test.suite()    )
    suite.addTest( pivpg_test.suite()    )
    suite.addTest( pivpost_test.suite()  )
    suite.addTest( pivsim_test.suite()   )
    suite.addTest( floutil_test.suite()  )
    suite.addTest( flotrace_test.suite() )
    suite.addTest( tlclib_test.suite()  )
    
    return suite

if __name__ == '__main__':
    print "----- RUNNING EXODUSII TESTS -----"
    import exodusII_test as ex2t
    print

    print "----- RUNNING SPIVET TESTS -----"
    unittest.TextTestRunner(verbosity=2).run(suite())
    print "======================================================================"

    print "EXODUSII Tests Run: %i, Failed %i" % (ex2t.ttcnt,ex2t.ftcnt)  


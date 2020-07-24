"""
Filename:  pivsim_test.py
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
    Runs various validation tests on the pivsim module.
"""

from spivet import pivlib
from numpy import *
import os, vtk, hashlib
from PIL import Image
import unittest, StringIO, sys
from os import path

#################################################################
#
def getPoints(vtkFileName):
    """
    Helper function to read the points array from a VTK file.
    """
    frdr = vtk.vtkPolyDataReader()
    frdr.SetFileName(vtkFileName)
    frdr.Update()

    da  = frdr.GetOutput().GetPoints().GetData()
    typ = da.GetDataType()

    # These come from the vtkConstants.py file from the VTK source.         
    if ( typ == 10 ):
        dtype = float32
    elif ( typ == 11 ):
        dtype = float64
    else:
        raise TypeError("Unkown VTK data type.")

    shape = [da.GetNumberOfTuples(),da.GetNumberOfComponents()]
    pts   = frombuffer(da,dtype=dtype).reshape(shape)
    return pts


#################################################################
#
class testSimObjectSetup(unittest.TestCase):
    def setUp(self):
        self.eps = 1.e-6
        self.dfp = "data"  # Input data file path.

        self.ofp = "test-output"  # Output file path
        if ( not path.exists(self.ofp) ):
            os.mkdir(self.ofp)

        # Redirect stdout for chatty functions.
        self.sobfr = StringIO.StringIO()
        sys.stdout = self.sobfr        

        # Create the SimObject.
        self.pos    = array((1,2,3))
        self.orient = array((4,5,6))
        
        self.so = pivlib.SimObject(self.pos,self.orient)

    def tearDown(self):
        self.sobfr.close()
        sys.stdout = sys.__stdout__   

    def testCreation(self):
        self.assertEqual( (self.pos == self.so.pos).all(), True )
        self.assertEqual( (self.orient == self.so.orient).all(), True )
        
    def testSetPosition(self):
        pos    = (-1.,0.,0.)
        self.so.setpos(pos)

        self.assertEqual( (pos == self.so.pos).all(), True )

    def testSetOrientation(self):
        orient = (pi/2.,pi/2.,pi/4.)

        self.so.setorient(orient)
        self.assertEqual( (orient == self.so.orient).all(), True )


#################################################################
#
class testSimObjectTransforms(unittest.TestCase):
    def setUp(self):
        self.eps = 1.e-6
        self.dfp = "data"  # Input data file path.

        self.ofp = "test-output"  # Output file path
        if ( not path.exists(self.ofp) ):
            os.mkdir(self.ofp)

        # Redirect stdout for chatty functions.
        self.sobfr = StringIO.StringIO()
        sys.stdout = self.sobfr        

        # Create the SimObject.
        self.pos    = array((-1.,0.,0.))
        self.orient = array((pi/2.,pi/2.,pi/4.))
        
        self.so = pivlib.SimObject(self.pos,self.orient)

        # Create some objects that will be used to test coordinate
        # transformations.
        self.pnt   = (1,0,0)
        self.ray   = pivlib.SimRay((0,0,0),self.pnt)
        self.krpnt = array((0,sqrt(2),sqrt(2)))  # Known.
        self.krvec = self.krpnt/2.               # Known.
         
    def tearDown(self):
        self.sobfr.close()
        sys.stdout = sys.__stdout__   
        
    def testPointParent2Local(self):
        rpnt = self.so.parent2local(self.pnt,0)
                
        v = rpnt -self.krpnt
        v = sqrt( (v*v).sum() )
        self.assertEqual( v < self.eps, True )

    def testVectorParent2Local(self):
        rvec = self.so.parent2local(self.pnt,True)
        
        v = rvec -self.krvec
        v = sqrt( (v*v).sum() )
        self.assertEqual( v < self.eps, True )

    def testRayParent2Local(self):
        rray = self.so.parent2local(self.ray,0)
        
        v1 = rray.source -self.krvec
        v2 = rray.head   -self.krvec
        v1 = sqrt( (v1*v1).sum() )
        v2 = sqrt( (v2*v2).sum() )
        self.assertEqual( ( v1 < self.eps ) and ( v2 < self.eps ), True ) 
        
    def testPointLocal2Parent(self):
        lpnt = self.so.parent2local(self.pnt,0)
        ppnt = self.so.local2parent(lpnt,0)
        
        v = ppnt -self.pnt
        v = sqrt( (v*v).sum() )
        self.assertEqual( v < self.eps, True )
        
    def testVectorLocal2Parent(self):
        lvec = self.so.parent2local(self.pnt,True)
        pvec = self.so.local2parent(lvec,True)
        
        v = pvec -self.pnt
        v = sqrt( (v*v).sum() )        
        self.assertEqual( v < self.eps, True )
 
 
#################################################################
#
class testSimRay(unittest.TestCase):
    def setUp(self):
        self.eps = 1.e-6
        self.dfp = "data"  # Input data file path.

        self.ofp = "test-output"  # Output file path
        if ( not path.exists(self.ofp) ):
            os.mkdir(self.ofp)

        # Redirect stdout for chatty functions.
        self.sobfr = StringIO.StringIO()
        sys.stdout = self.sobfr        

        # Create the SimRay.
        ray = pivlib.SimRay((0.,0.,0.),(1.,0.,0.))
        ray.addSegment(1.)
        ray.changeHeading((4.,0.,3.))
        ray.addSegment(5.)
        ray.changeHeading((1.,0.,0.))
        ray.addSegment(1.)
        
        self.ray = ray
        
        self.ksource = (6.,0.,3.)
         
    def tearDown(self):
        self.sobfr.close()
        sys.stdout = sys.__stdout__   
        
    def testSegmentCount(self):
        self.assertEqual(self.ray.segments, 3)
        
    def testSource(self):
        v = abs( self.ray.source -self.ksource )
        v = sqrt( (v*v).sum() )        
        self.assertEqual( v < self.eps, True )

    def testVTKOutput(self):
        pd = vtk.vtkPolyData()
        pd.Allocate(1,1)
        pd.SetPoints(vtk.vtkPoints())
        pd  = pivlib.getRayVTKRep(self.ray,pd)
        pts = pd.GetPoints()
        
        self.assertEqual(pts.GetNumberOfPoints(),4)
        
        pflg = True
        kpts = array([[0.,0.,0.],[1.,0.,0.],[5.,0.,3.],[6.,0.,3.]])
        for i in range(4):
            v = abs( pts.GetPoint(i) -kpts[i][::-1] )
            if ( ( v < self.eps ).all() ):
                continue
            else:
                pflg = False
        self.assertEqual(pflg,True)


#################################################################
#
class testSimCylindricalSurface(unittest.TestCase):
    def setUp(self):
        self.eps = 1.e-6
        self.dfp = "data"  # Input data file path.

        self.ofp = "test-output"  # Output file path
        if ( not path.exists(self.ofp) ):
            os.mkdir(self.ofp)

        # Redirect stdout for chatty functions.
        self.sobfr = StringIO.StringIO()
        sys.stdout = self.sobfr        

        pos     = (0.,0.,0.)
        orient  = (pi/2.,pi/2.,0)   # c0 along x-axis, c2 along y-axis.
        prm     = [3,5]
        cs      = pivlib.SimCylindricalSurf(pos,orient,prm) 
        self.cs = cs
                 
    def tearDown(self):
        self.sobfr.close()
        sys.stdout = sys.__stdout__   
        
    def testIntersection1(self):
        ray = pivlib.SimRay((0.,0.,-4.), (0.,0.,4.))
        kt  = 1
        tt  = self.cs.intersect(ray)[0]
        v   = abs(tt -kt)
        pnt = ray.point(tt)
        kn  = (0.,0.,-1.)
        tn  = self.cs.normal(pnt) 
        nv  = 1. -dot(tn,kn) 

        self.assertEqual(v < self.eps, True)   # Hit
        self.assertEqual(nv < self.eps, True)  # Normal

    def testIntersection2(self):
        ray = pivlib.SimRay((0.,0.,0.),(0.,0.,4.))
        kt  = 3
        tt  = self.cs.intersect(ray)[0]
        v   = abs(tt -kt)
        pnt = ray.point(tt)
        kn  = (0.,0.,1.)
        tn  = self.cs.normal(pnt) 
        nv  = 1. -dot(tn,kn)

        self.assertEqual(v < self.eps, True)   # Hit
        self.assertEqual(nv < self.eps, True)  # Normal

    def testIntersection3(self):
        ray = pivlib.SimRay((0.,0.,0.5),(0.,0.,-4.))
        kt  = 3.5
        tt  = self.cs.intersect(ray)[0]
        v   = abs(tt -kt)

        self.assertEqual(v < self.eps, True)   # Hit

    def testIntersection4(self):
        ray = pivlib.SimRay((0.,-3.,-4.), (0.,0.,4.))
        kt  = 4
        tt  = self.cs.intersect(ray)[0]
        v   = abs(tt -kt)

        self.assertEqual(v < self.eps, True)   # Hit

    def testIntersection5(self):
        ray = pivlib.SimRay((0.,-4.,0.), (3.,4.,0.))
        kt  = 1.25
        tt  = self.cs.intersect(ray)[0]
        v   = abs(tt -kt)

        self.assertEqual(v < self.eps, True)   # Hit

    def testMiss(self):
        ray = pivlib.SimRay((0.,-3.,-4.), (0.,-self.eps,4.))
        tt  = self.cs.intersect(ray)

        self.assertEqual(tt,None)  # Miss

    def testParallelMiss(self):
        ray = pivlib.SimRay((0.,-2.,0.), (3.,0.,0.))
        tt  = self.cs.intersect(ray)

        self.assertEqual(tt,None)  # Miss


#################################################################
#
class testSimRectPlanarSurface(unittest.TestCase):
    def setUp(self):
        self.eps = 1.e-6
        self.dfp = "data"  # Input data file path.

        self.ofp = "test-output"  # Output file path
        if ( not path.exists(self.ofp) ):
            os.mkdir(self.ofp)

        # Redirect stdout for chatty functions.
        self.sobfr = StringIO.StringIO()
        sys.stdout = self.sobfr        

        pos    = (0.,0.,0.)
        orient = (pi/2.,pi/2.,0)   # c0 along x-axis, c2 along y-axis.
        prm    = [3,5]
        ps     = pivlib.SimRectPlanarSurf(pos,orient,prm) 
        self.ps = ps
                 
    def tearDown(self):
        self.sobfr.close()
        sys.stdout = sys.__stdout__   
        
    def testIntersection1(self):
        ray = pivlib.SimRay((4.,1.,0.),(-4.,0.,3.))
        kt  = 5
        tt  = self.ps.intersect(ray)[0]
        v   = abs(tt -kt)

        self.assertEqual(v < self.eps, True)   # Hit

    def testIntersection2(self):
        ray = pivlib.SimRay((-4.,1.,0.),(4.,0.,3.))
        kt  = 5
        tt  = self.ps.intersect(ray)[0]
        v   = abs(tt -kt)

        self.assertEqual(v < self.eps, True)   # Hit
        
    def testIntersection3(self):
        ray = pivlib.SimRay((4.,-2.,-1.),(-4.,3.,0.))
        kt  = 5
        tt  = self.ps.intersect(ray)[0]
        v   = abs(tt -kt)

        self.assertEqual(v < self.eps, True)   # Hit

    def testMiss1(self):
        ray = pivlib.SimRay((4.,1.,0.),(4.,0.,3.))
        tt  = self.ps.intersect(ray)

        self.assertEqual(tt,None)  # Miss

    def testMiss2(self):
        ray = pivlib.SimRay((4.,1.,0.),(-4.,0.,5.+self.eps))
        tt  = self.ps.intersect(ray)

        self.assertEqual(tt,None)  # Miss

    def testParallelMiss(self):        
        ray = pivlib.SimRay((4.,1.,0.),(0.,0.,5.))
        tt  = self.ps.intersect(ray)
        
        self.assertEqual(tt,None)  # Miss


#################################################################
#
class testSimCircPlanarSurface(unittest.TestCase):
    def setUp(self):
        self.eps = 1.e-6
        self.dfp = "data"  # Input data file path.

        self.ofp = "test-output"  # Output file path
        if ( not path.exists(self.ofp) ):
            os.mkdir(self.ofp)

        # Redirect stdout for chatty functions.
        self.sobfr = StringIO.StringIO()
        sys.stdout = self.sobfr        

        pos    = (0.,0.,0.)
        orient = (pi/2.,pi/2.,0)   # c0 along x-axis, c2 along y-axis.
        prm    = 5
        ps     = pivlib.SimCircPlanarSurf(pos,orient,prm) 

        self.ps = ps
                 
    def tearDown(self):
        self.sobfr.close()
        sys.stdout = sys.__stdout__   
        
    def testIntersection1(self):
        ray = pivlib.SimRay((4.,1.,0.),(-4.,0.,3.))
        kt  = 5
        tt  = self.ps.intersect(ray)[0]
        v   = abs(tt -kt)
        
        self.assertEqual(v < self.eps, True)   # Hit

    def testIntersection2(self):
        ray = pivlib.SimRay((-4.,1.,0.),(4.,0.,3.))
        kt  = 5
        tt  = self.ps.intersect(ray)[0]
        v   = abs(tt -kt)

        self.assertEqual(v < self.eps, True)   # Hit

    def testIntersection3(self):
        ray = pivlib.SimRay((4.,-2.,-1.),(-4.,3.,0.))
        kt  = 5
        tt  = self.ps.intersect(ray)[0]
        v   = abs(tt -kt)

        self.assertEqual(v < self.eps, True)   # Hit

    def testMiss1(self):
        ray = pivlib.SimRay((4.,1.,0.),(4.,0.,3.))
        tt  = self.ps.intersect(ray)

        self.assertEqual(tt,None)  # Miss

    def testMiss2(self):
        ray = pivlib.SimRay((4.,1.,0.),(-4.,0.,5.))
        tt  = self.ps.intersect(ray)

        self.assertEqual(tt,None)  # Miss

    def testMiss3(self):
        ray = pivlib.SimRay((4.,0.,0.),(-4.,0.,5.+self.eps))
        tt  = self.ps.intersect(ray)

        self.assertEqual(tt,None)  # Miss

    def testParallelMiss(self):
        ray = pivlib.SimRay((4.,1.,0.),(0.,0.,5.))
        tt  = self.ps.intersect(ray)

        self.assertEqual(tt,None)  # Miss


#################################################################
#
class testSimLeafNode(unittest.TestCase):
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

    def testSphereLeafBounds(self):
        pos     = (-7.,10.,-7.)
        obj     = pivlib.SimSphere(pos,(0.,0.,0.),1.,1.)
        kbounds = obj.lbounds +pos
        leaf    = pivlib.SimLeafNode(obj)
        v       = leaf.bounds -kbounds
        self.assertEqual( ( abs(v) < self.eps ).all(), True )

    def testCylinderLeafBounds(self):
        pos     = (0.,-7.5,2.5)
        orient  = (0.,pi/4.,0.)
        pc0     = 2.5*cos(arccos(3./5.) -pi/4.)
        pc1     = 2.5*sin(arccos(3./5.) +pi/4.) 
        pc2     = 1.5
        obj     = pivlib.SimCylinder(pos,orient,[1.5,2.],1.)
        kbounds = array([[-pc0,-pc1,-pc2],
                         [-pc0,-pc1, pc2],
                         [-pc0, pc1, pc2],
                         [-pc0, pc1,-pc2],
                         [ pc0,-pc1,-pc2],
                         [ pc0,-pc1, pc2],
                         [ pc0, pc1, pc2],
                         [ pc0, pc1,-pc2]] ) 
        kbounds = kbounds +pos
        leaf    = pivlib.SimLeafNode(obj)
        v       = leaf.bounds -kbounds
        self.assertEqual( ( abs(v) < self.eps ).all(), True )


#################################################################
#
class testSimOctree(unittest.TestCase):
    def setUp(self):
        self.eps = 1.e-6
        self.dfp = "data"  # Input data file path.

        self.ofp = "test-output"  # Output file path
        if ( not path.exists(self.ofp) ):
            os.mkdir(self.ofp)

        # Redirect stdout for chatty functions.
#       self.sobfr = StringIO.StringIO()
#       sys.stdout = self.sobfr        

        # Setup the test.
        ot = pivlib.SimOctree((10.,15.,10.),3)
        
        pos     = (-7.,10.,-7.)
        obj     = pivlib.SimSphere(pos,(0.,0.,0.),1.,1.)
        leaf    = pivlib.SimLeafNode(obj)        
        ot.addLeaf(leaf)  # LN0
        
        pos     = (0.,0.,0.)
        obj     = pivlib.SimSphere(pos,(0.,0.,0.),1.,1.)
        leaf    = pivlib.SimLeafNode(obj)
        ot.addLeaf(leaf)  #LN1
        
        pos     = (0.,-7.5,2.5)
        orient  = (0.,pi/4.,0.)
        obj     = pivlib.SimCylinder(pos,orient,[1.5,2.],1.)
        leaf    = pivlib.SimLeafNode(obj)
        ot.addLeaf(leaf)  # LN2
        
        pos     = (-5.,-7.5,5.)
        obj     = pivlib.SimSphere(pos,(0.,0.,0.),1.,1.)
        leaf    = pivlib.SimLeafNode(obj)
        ot.addLeaf(leaf)  # LN3
        
        pos     = (-3.5,-10.,1.5)
        obj     = pivlib.SimSphere(pos,(0.,0.,0.),1.,1.)
        leaf    = pivlib.SimLeafNode(obj)
        ot.addLeaf(leaf)  # LN4
        
        pos     = (-1.5,-10.,1.5)
        obj     = pivlib.SimSphere(pos,(0.,0.,0.),1.,1.)
        leaf    = pivlib.SimLeafNode(obj)
        ot.addLeaf(leaf)  # LN5

        self.ot = ot

    def tearDown(self):
#       self.sobfr.close()
        sys.stdout = sys.__stdout__

    def testGraph(self):
        ofname = "%s/graph" % self.ofp
        kfname = "%s/graph-known" % self.dfp
        self.ot.dump2graphviz(ofname,None)
        fh = open(ofname,'r')
        og = fh.read()
        fh.close()
        fh = open(kfname,'r')
        kg = fh.read()
        fh.close()
        self.assertEqual(og,kg)

    def testOctreeVTKINode(self):
        ofname = "%s/octree" % self.ofp
	print "I am here: %s" % (ofname)
        self.ot.dump2vtk(ofname,None)
        
        kfname = "%s/octree-inode-known.vtk" % self.dfp
        kpts   = getPoints(kfname)
    
        tpts = getPoints("%s-INODE.vtk" % ofname)
        
        d = abs(tpts -kpts) < self.eps
        self.assertTrue( d.all() )

    def testoctreeVTKLNode(self):
        ofname = "%s/octree" % self.ofp
        self.ot.dump2vtk(ofname)

        kfname = "%s/octree-lnode-known.vtk" % self.dfp
        kpts   = getPoints(kfname)
    
        tpts = getPoints("%s-LNODE.vtk" % ofname)
        
        d = abs(tpts -kpts) < self.eps
        self.assertTrue( d.all() )


#################################################################
#
class testSimRefractiveObject(unittest.TestCase):
    def setUp(self):
        self.eps = 1.e-6
        self.dfp = "data"  # Input data file path.

        self.ofp = "test-output"  # Output file path
        if ( not path.exists(self.ofp) ):
            os.mkdir(self.ofp)

        # Redirect stdout for chatty functions.
        self.sobfr = StringIO.StringIO()
        sys.stdout = self.sobfr        

        # Create test object.
        pos    = (0.,0.,0.)
        orient = (0.,0.,0.)
        prm    = [5,5]
        rs     = pivlib.SimRectPlanarSurf(pos,orient,prm)
        
        self.kn     = 1.49
        self.pos    = array((1,2,3))
        self.orient = array((4,5,6))
        self.ro = pivlib.SimRefractiveObject(self.pos,
                                             self.orient,
                                             self.kn,
                                             [rs])

    def tearDown(self):
        self.sobfr.close()
        sys.stdout = sys.__stdout__

    def testCreate(self):
        self.assertEqual( (self.ro.pos    == self.pos   ).all(), True)
        self.assertEqual( (self.ro.orient == self.orient).all(), True)

        self.assertEqual(self.ro.n, self.kn)

    def testSetPosition(self):
        pos    = (-1.,0.,0.)
        self.ro.setpos(pos)

        self.assertEqual( (pos == self.ro.pos).all(), True )

    def testSetOrientation(self):
        orient = (pi/2.,pi/2.,pi/4.)
        self.ro.setorient(orient)

        self.assertEqual( (orient == self.ro.orient).all(), True )


#################################################################
#
class testSimCamera(unittest.TestCase):
    def setUp(self):
        self.eps = 1.e-6
        self.dfp = "data"  # Input data file path.

        self.ofp = "test-output"  # Output file path
        if ( not path.exists(self.ofp) ):
            os.mkdir(self.ofp)

        # Redirect stdout for chatty functions.
        self.sobfr = StringIO.StringIO()
        sys.stdout = self.sobfr        

        # Create test object.
        pos    = (0.,0.,0.)
        orient = (0.,0.,0.)
        prm    = (4.,3,3.,1,1.)
        sc     = pivlib.SimCamera(pos,orient,prm)
        
        self.ksource = (0.,0.,0.)
        self.kheadc  = (1.,0.,0.)
        self.kheadle = array((4.,-3.,0.))/5.
        self.kheadre = array((4.,3.,0.))/5.
        self.rays    = sc.initrays()

    def tearDown(self):
        self.sobfr.close()
        sys.stdout = sys.__stdout__

    def testRayCount(self):
        self.assertEqual(len(self.rays),3)

    def testSource(self):
        tst = True
        for i in range(3):
            vs  = abs(self.ksource -self.rays[i].source) 
            tst = tst*( (vs < self.eps).all() )

        self.assertEqual(tst,True)
        
    def testLeftRayHeading(self):
        vh = abs(self.kheadle -self.rays[0].head)
        
        self.assertEqual( (vh < self.eps).all(), True )

    def testCenterRayHeading(self):
        vh = abs(self.kheadc -self.rays[1].head)
        
        self.assertEqual( (vh < self.eps).all(), True )

    def testRightRayHeading(self):
        vh = abs(self.kheadre -self.rays[2].head)
        
        self.assertEqual( (vh < self.eps).all(), True )
        

#################################################################
#
class testSimLight(unittest.TestCase):
    def setUp(self):
        self.eps = 1.e-6
        self.dfp = "data"  # Input data file path.

        self.ofp = "test-output"  # Output file path
        if ( not path.exists(self.ofp) ):
            os.mkdir(self.ofp)

        # Redirect stdout for chatty functions.
        self.sobfr = StringIO.StringIO()
        sys.stdout = self.sobfr        

        # Create test object.
        orient  = (pi/2.,pi/2.,0.)  # c0 along x-axis, c2 along y-axis.
        self.lg = pivlib.SimLight(orient)

        self.klsource = (0.,0.,0.)
        self.klhead   = (1.,0.,0.)

        self.kisource = (0.,0.,0.)
        self.kihead = (0.,0.,1.)

    def tearDown(self):
        self.sobfr.close()
        sys.stdout = sys.__stdout__
        
    def testIRaySource(self):
        v = abs( self.lg.iray.source -self.kisource )
        
        self.assertEqual( ( v < self.eps ).all(), True )

    def testIRayHeading(self):
        v = abs( self.lg.iray.head -self.kihead )
        
        self.assertEqual( ( v < self.eps ).all(), True )

    def testLocalIRaySource(self):
        v = abs( self.lg.liray.source -self.klsource )

        self.assertEqual( ( v < self.eps ).all(), True )


    def testLocalIRayHeading(self):
        v = abs( self.lg.liray.head -self.klhead )

        self.assertEqual( ( v < self.eps ).all(), True )
        

#################################################################
#
class testSimEnv(unittest.TestCase):
    def setUp(self):
        self.eps = 1.e-6
        self.dfp = "data"  # Input data file path.

        self.ofp = "test-output"  # Output file path
        if ( not path.exists(self.ofp) ):
            os.mkdir(self.ofp)

        # Redirect stdout for chatty functions.
        self.sobfr = StringIO.StringIO()
        sys.stdout = self.sobfr        

        # Create test object.
        pos     = (-5.,0.,0.)
        orient  = (0.,0.,0.)  
        prm     = [5.,5.,5.]
        self.ln = 1.49
        self.ro = pivlib.SimRectangle(pos,orient,prm,self.ln)
        
        prm    = (10.,15.,10.)
        self.n = 1.
        se     = pivlib.SimEnv(prm,self.n)
        se.addObject(self.ro)
        
        self.se = se

    def tearDown(self):
        self.sobfr.close()
        sys.stdout = sys.__stdout__
        
    def testRefraction1(self):
        dmy   = sqrt(3.)/(2.*self.ln)
        khead = array([-sqrt(1. -dmy*dmy),0.,dmy])
        ray   = pivlib.SimRay((0.5,0.,0.),(-0.5,0.,sqrt(3.)/2.))
        lray  = self.ro.parent2local(ray,0)
        insc  = self.ro.intersect(lray)
        rray  = self.se.refract(self.n,self.ln,insc[0])
        v     = abs( rray.head -khead )
        
        self.assertEqual(len(insc),2)          # Number of intersections.
        self.assertFalse(insc[0].exflg)        # Exit.
        self.assertTrue((v < self.eps).all())  
        
    def testRefraction2(self):
        dmy   = sqrt(3.)/(2.*self.ln)
        khead = array([-sqrt(1. -dmy*dmy),0.,-dmy])
        ray   = pivlib.SimRay((0.5,0.,0.),(-0.5,0.,-sqrt(3.)/2.))
        lray  = self.ro.parent2local(ray,0)
        insc  = self.ro.intersect(lray)
        rray  = self.se.refract(self.n,self.ln,insc[0])
        v     = abs( rray.head -khead )

        self.assertEqual(len(insc),2)          # Number of intersections.
        self.assertFalse(insc[0].exflg)        # Exit.
        self.assertTrue((v < self.eps).all())  

    def testNormalIncidence(self):
        khead = array([-1.,0.,0.])
        ray   = pivlib.SimRay((0.5,0.,0.),(-1.,0.,0.))
        lray  = self.ro.parent2local(ray,0)
        insc  = self.ro.intersect(lray)
        rray  = self.se.refract(self.n,self.ln,insc[0])
        v     = abs( rray.head -khead )

        self.assertEqual(len(insc),2)          # Number of intersections.
        self.assertFalse(insc[0].exflg)        # Exit.
        self.assertTrue((v < self.eps).all())  

    def testTotalInternalReflection(self):
        khead = array([0.5,0.,sqrt(3.)/2.])
        ray   = pivlib.SimRay((0.5,0.,0.),(-0.5,0.,sqrt(3.)/2.))
        lray  = self.ro.parent2local(ray,0)
        insc  = self.ro.intersect(lray)
        rray  = self.se.refract(1.8,self.ln,insc[0])
        v     = abs( rray.head -khead )

        self.assertEqual(len(insc),2)          # Number of intersections.
        self.assertFalse(insc[0].exflg)        # Exit.
        self.assertTrue((v < self.eps).all())  

    def testExitingRefraction1(self):
        dmy   = sqrt(3.)/(2.*self.ln)
        khead = array([sqrt(1. -dmy*dmy),0.,dmy])
        ray   = pivlib.SimRay((-0.5,0.,0.),(0.5,0.,sqrt(3.)/2.))
        lray  = self.ro.parent2local(ray,0)
        insc  = self.ro.intersect(lray)
        rray  = self.se.refract(self.n,self.ln,insc[0])
        v     = abs( rray.head -khead )

        self.assertEqual(len(insc),1)          # Number of intersections.
        self.assertTrue(insc[0].exflg)         # Exit.
        self.assertTrue((v < self.eps).all())  

    def testExitingRefraction2(self):
        dmy   = 0.5*self.ln
        khead = array([sqrt(1. -dmy*dmy),0.,dmy])
        ray   = pivlib.SimRay((-sqrt(3.)/2.,0.,0.),(sqrt(3.)/2.,0.,0.5))
        lray  = self.ro.parent2local(ray,0)
        insc  = self.ro.intersect(lray)
        rray  = self.se.refract(self.ln,self.n,insc[0])
        v     = abs( rray.head -khead )

        self.assertEqual(len(insc),1)          # Number of intersections.
        self.assertTrue(insc[0].exflg)         # Exit.
        self.assertTrue((v < self.eps).all())  


#################################################################
#
class testTraceRectangle(unittest.TestCase):
    def setUp(self):
        self.eps = 1.e-6
        self.dfp = "data"  # Input data file path.

        self.ofp = "test-output"  # Output file path
        if ( not path.exists(self.ofp) ):
            os.mkdir(self.ofp)

        # Redirect stdout for chatty functions.
        self.sobfr = StringIO.StringIO()
        sys.stdout = self.sobfr        

        # Create test object.
        pos    = (-9.,0.,0.)
        orient = (0.,0.,0.)
        prm    = (8.,1,1.,3,6.)
        sc     = pivlib.SimCamera(pos,orient,prm)
        
        pos    = (2.,0.,0.)
        orient = (pi/6.,5*pi/6.,0.)
        prm    = [2.,10.,10.]
        ln     = 1.49
        ro     = pivlib.SimRectangle(pos,orient,prm,ln)
        
        prm = (10.,15.,10.)
        n   = 1.
        se  = pivlib.SimEnv(prm,n)
        se.addObject(ro)
        se.addCamera(sc)
        se.image()

        self.bofname = "%s/simenv-rect" % self.ofp
        se.dump2vtk(self.bofname)

    def tearDown(self):
        self.sobfr.close()
        sys.stdout = sys.__stdout__
        
    def testOutput(self):
        ofname = "%s-RAYS-C0.vtk" % self.bofname
        tpts   = getPoints(ofname)
        kfname = "%s/simenv-rect-rays-known.vtk" % self.dfp
        kpts   = getPoints(kfname)        

        d = abs(tpts -kpts) < self.eps
        self.assertTrue( d.all(), "RAYS" )

        ofname = "%s-LNODE.vtk" % self.bofname
        tpts   = getPoints(ofname)
        kfname = "%s/simenv-rect-lnode-known.vtk" % self.dfp
        kpts   = getPoints(kfname)        

        d = abs(tpts -kpts) < self.eps
        self.assertTrue( d.all(), "LNODE" )
        

#################################################################
#
class testTraceCylinder(unittest.TestCase):
    def setUp(self):
        self.eps = 1.e-6
        self.dfp = "data"  # Input data file path.

        self.ofp = "test-output"  # Output file path
        if ( not path.exists(self.ofp) ):
            os.mkdir(self.ofp)

        # Redirect stdout for chatty functions.
        self.sobfr = StringIO.StringIO()
        sys.stdout = self.sobfr        

        # Create test object.
        pos    = (-9.,0.,0.)
        orient = (0.,0.,0.)
        prm    = (9.,1,1.,3,3./sqrt(2.))
        sc     = pivlib.SimCamera(pos,orient,prm)
        
        pos    = (3./sqrt(2.),0.,0.)
        orient = (0.,pi/2.,0.)  
        prm    = [3.,7.5]
        ln     = 1.49
        ro     = pivlib.SimCylinder(pos,orient,prm,ln)
        
        prm = (10.,15.,10.)
        n   = 1.
        se  = pivlib.SimEnv(prm,n)
        se.addObject(ro)
        se.addCamera(sc)
        se.image()
        
        self.bofname = "%s/simenv-cyl" % self.ofp
        se.dump2vtk(self.bofname)

    def tearDown(self):
        self.sobfr.close()
        sys.stdout = sys.__stdout__
        
    def testOutput(self):
        ofname = "%s-RAYS-C0.vtk" % self.bofname
        tpts   = getPoints(ofname)
        kfname = "%s/simenv-cyl-rays-known.vtk" % self.dfp
        kpts   = getPoints(kfname)        

        d = abs(tpts -kpts) < self.eps
        self.assertTrue( d.all(), "RAYS" )

        ofname = "%s-LNODE.vtk" % self.bofname
        tpts   = getPoints(ofname)
        kfname = "%s/simenv-cyl-lnode-known.vtk" % self.dfp
        kpts   = getPoints(kfname)        

        d = abs(tpts -kpts) < self.eps
        self.assertTrue( d.all(), "LNODE" )
        

#################################################################
#
class testTraceBitmapRectangle(unittest.TestCase):
    def setUp(self):
        self.eps = 1.e-6
        self.dfp = "data"  # Input data file path.

        self.ofp = "test-output"  # Output file path
        if ( not path.exists(self.ofp) ):
            os.mkdir(self.ofp)

        # Redirect stdout for chatty functions.
        self.sobfr = StringIO.StringIO()
        sys.stdout = self.sobfr        

        # Create test object.
        pos    = (-1000.,0.,0.)
        orient = (0.,0.,0.)
        prm    = (15.,768,4.65E-3,50,4.65E-3)
        sc     = pivlib.SimCamera(pos,orient,prm)
        
        pos    = (5.,0.,0.)
        orient = (pi/6.,5.*pi/6.,0.)
        prm    = (10.,100.,100.)
        ln     = None
        ro     = pivlib.SimRectangle(pos,orient,prm,ln)
        
        bitmap = ones((100,100),dtype=float)
        bitmap[10:60,50:80] = 0.
        ro.setBitmap(bitmap)
        
        prm = (1200.,500.,500.)
        n   = 1.
        se  = pivlib.SimEnv(prm,n)
        se.addObject(ro)
        se.addCamera(sc)
        se.image()
        
        self.ofname  = "%s/surface-img-render.png" % self.ofp
        self.kfname  = "%s/surface-img-render-known.png" % self.dfp
        pivlib.imwrite(sc.bitmap,self.ofname,vmin=0.,vmax=1.)

    def tearDown(self):
        self.sobfr.close()
        sys.stdout = sys.__stdout__
        
    def testImagesMatch(self):
        fh = open(self.ofname,'r')
        og = fh.read()
        fh.close()
        fh = open(self.kfname,'r')
        kg = fh.read()
        fh.close()
        
        tm = hashlib.md5()
        tm.update(og)
        km = hashlib.md5()
        km.update(kg)
        
        self.assertEqual(tm.hexdigest(),km.hexdigest())


#################################################################
#        
class testTraceSurfaceRender(unittest.TestCase):
    def setUp(self):
        self.eps = 1.e-6
        self.dfp = "data"  # Input data file path.

        self.ofp = "test-output"  # Output file path
        if ( not path.exists(self.ofp) ):
            os.mkdir(self.ofp)

        # Redirect stdout for chatty functions.
        self.sobfr = StringIO.StringIO()
        sys.stdout = self.sobfr        

        # Create test object.
        pos    = (-1000.,0.,0.)
        orient = (0.,0.,0.)
        prm    = (15.,20,4.65E-3,80,4.65E-3)
        sc     = pivlib.SimCamera(pos,orient,prm)
        
        pos    = (2.,0.,0.)
        orient = (0.,pi/2.,0.)  
        prm    = [20.,100.]
        ln     = None
        ro     = pivlib.SimCylinder(pos,orient,prm,ln)
        
        prm = (1200.,500.,500.)
        n   = 1.
        se  = pivlib.SimEnv(prm,n)
        se.addObject(ro)
        se.addCamera(sc)
        se.image()
        
        self.ofname  = "%s/surface-render.png" % self.ofp
        self.kfname  = "%s/surface-render-known.png" % self.dfp
        pivlib.imwrite(sc.bitmap,self.ofname,vmin=0.,vmax=1.)

    def tearDown(self):
        self.sobfr.close()
        sys.stdout = sys.__stdout__
        
    def testImagesMatch(self):
        fh = open(self.ofname,'r')
        og = fh.read()
        fh.close()
        fh = open(self.kfname,'r')
        kg = fh.read()
        fh.close()
        
        tm = hashlib.md5()
        tm.update(og)
        km = hashlib.md5()
        km.update(kg)
        
        self.assertEqual(tm.hexdigest(),km.hexdigest())


def suite():
    suite = unittest.TestSuite()
    suite.addTest( unittest.makeSuite( testSimObjectSetup        ) )
    suite.addTest( unittest.makeSuite( testSimObjectTransforms   ) )
    suite.addTest( unittest.makeSuite( testSimRay                ) )
    suite.addTest( unittest.makeSuite( testSimCylindricalSurface ) )
    suite.addTest( unittest.makeSuite( testSimRectPlanarSurface  ) )
    suite.addTest( unittest.makeSuite( testSimCircPlanarSurface  ) )
    suite.addTest( unittest.makeSuite( testSimLeafNode           ) )
    suite.addTest( unittest.makeSuite( testSimOctree             ) )
    suite.addTest( unittest.makeSuite( testSimRefractiveObject   ) )
    suite.addTest( unittest.makeSuite( testSimCamera             ) )
    suite.addTest( unittest.makeSuite( testSimLight              ) )
    suite.addTest( unittest.makeSuite( testSimEnv                ) )
    suite.addTest( unittest.makeSuite( testTraceRectangle        ) )
    suite.addTest( unittest.makeSuite( testTraceCylinder         ) )
    suite.addTest( unittest.makeSuite( testTraceBitmapRectangle  ) )
    suite.addTest( unittest.makeSuite( testTraceSurfaceRender    ) )
        
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(suite())


"""
Filename:  pivsim.py
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
  pivsim provides raytracing facilities to simulate the PIV
  environment.  Each simulation constructs a grayscale image of 
  the simulation environment for any number of cameras.  Like
  all other PivLIB images, the intensity of each pixel varies
  between 0.0 .. 1.0.

  Simulation is performed by launching rays from a pinhole camera.
  A ray corresponding to each pixel in the camera is propagated 
  through the environment until the ray either strikes an environment
  boundary (at which point the corresponding pixel intesity is set to 0.0), 
  or hits an imageable object.  The ray will take its intensity value
  from the first imageable object it strikes (so opaque objects do
  occlude each other).

  A simulation must consist of the following entities:
      1) One SimEnv object.  The SimEnv object represent the laboratory
      environment.  The SimEnv, just like all other refractive objects,
      has a refractive index, which should generally be set 1.0 (unless
      the camera is immersed in a fluid).  All imageable objects must
      lie within the perimeter of the SimEnv.

      2) At least one SimCamera object.  The SimCamera represents the real 
      laboratory cameras and should be positioned as such.  A SimCamera is 
      a simple pinhole camera, and subsequently no attempt is made to model 
      the optics of the camera lens.  The distortions caused by lenses are
      well beyond the sophistication of the pivsim module.  More elaborate
      tools are mentioned below.

      3) Any number of SimSpheres, SimCylinders, or SimRectangles.  These
      three primitive solids are all derived from a more primitive 
      SimRefractiveObject type.  The laboratory environment is meant to be
      composed of SimRefractiveObjects, with tracers represented by opaque
      SimSpheres and so on.  

      A SimRefractiveObject with a numerical refractive index will behave 
      just like other transparent objects, and refract the ray (total 
      internal reflection is supported).  At present, only objects with a 
      uniform, constant refractive index are permitted.  SimRefractiveObjects 
      with an refractive index equal to None are imageable objects 
      (ie, opaque).    

  pivsim does not place any constraints on the relative orientation
  of SimRefractiveObjects within the SimEnv.  Actually, the SimEnv object
  is a SimRefractiveObject itself, so objects can and should be nested 
  within one another.  However, the user must exercise caution here as 
  pivsim is not a full-fledged CAD package, so its ability to handle nested 
  objects is limited.  As long as one SimRefractiveObject is completely 
  contained within another SimRefractiveObject, pivsim will function 
  correctly.  The following scenario demonstrates an object configuration 
  that will generate spurious results.

                      ------  SimRectangle 1, n = 1.3
                      |    |
                   ------  |
                   |  |-|---
                   |    |
                   ------  SimRectangle 2, n = 1.5

  In the above example, two SimRectangle objects intersect, but one is not
  fully contained within the other.  Because pivsim only uses three
  primitive 3D solid shapes (sphere, rectangle, cylinder), no facility
  is provided to handle abritrary surface configurations.  As a result,
  the above configuration is ambiguous.  It's not clear whether the union
  section (the common volume between the two SimRectangles) should be its
  own separate material with a unique refractive index, or if one of the
  rectangles fits inside the other like two puzzle pieces.  If arbitrary 
  solids are required with a full complement of boolean operations, then 
  a more sophisticated package should be used.  Also note that objects 
  which are tangent to each other will generate spurious results.

               -----------
               | n1 | n2 |
               |    |    |
               -----------

  In this example, two rectangles of differing indices butt up against each 
  other.  pivsim will not choose the correct refractive indices.  The
  preferred setup for the example would be one of two approaches:
      a) let the object with refractive index n1 (or n2) completely enclose 
         the other
      b) include a very small (ie, approximately 1E-6) air gap between the two 
         objects so that the bounding interfaces can be correctly determined.

  The SimEnv object defines a world coordinate system (z,y,x) with respect 
  to which all other objects are oriented.  A given object is placed in 
  the environment by specifying the origin location (pos) and orientation 
  (orient) of the object's local coordinate system.  The object's local 
  coordinate system is spanned by the c0, c1, and c2-axes.  The pos variable
  is straightforward and represents the translation vector from the SimEnv
  origin to the origin of the object's local coordinate system.  The orient
  variable specifies three Euler angles necessary to rotate the object's
  local coordinate system as desired.  The object will initially be inserted
  with its c0-axis parallel to the z-axis, the c1-axis parallel to y, and
  the c2-axis parallel to x.  The Euler angles phi, theta, and psi then
  define rotations about the object's c0, c2, c0 axes in that order.  Each
  rotation is applied about the object's current local coordinates, so if
  phi is non-zero then theta is about the new c2 (ie, not about the global 
  x).  For a discussion of Euler angles, consult Goldstein:1980.

  NOTE: As will all PivLIB functions, coordinate arrays are specified in 
  the order [z,y,x] or [c0,c1,c2].

  NOTE:  If the user needs to investigate variable index materials, 
  lens effects, radiometric performance (tranmittance, reflectance, overall 
  throughput, etc), or just more complex geometries with arbitrary surfaces, 
  a more robust tool should be used.  Several vendors (eg. Lambda Research, 
  ZEMAX, Optical Research Associates) provide very sophisticated alternatives.

  pivsim stores all SimRefractiveObjects in an octree to speed raytracing.  
  More details on the internals can be found in the octree classes SimOctree,
  SimNode, SimInternalNode, and SimLeafNode.
 
Contents:
  Base and Internal Classes (classes that the user doesn't need):
    class SimRay
    class SimIntersection
    class SimObject
    class SimSurf(SimObject)
    class SimCylindricalSurf(SimSurf)
    class SimCircPlanarSurf(SimSurf)
    class SimRectPlanarSurf(SimSurf)
    class SimSphericalSurf(SimSurf)
    class SimRefractiveObject(SimObject)
    class SimNode(SimRectangle)
    class SimInternalNode(SimNode)
    class SimLeafNode(SimNode)
    class SimOctree(SimNode)

  User-level classes:
    class SimCylinder(SimRefractiveObject)
    class SimRectangle(SimRefractiveObject)
    class SimSphere(SimRectangle)
    class SimLight(SimObject)
    class SimCamera(SimObject)
    class SimEnv(SimRectangle)
    
  Utility Functions:
    getRayVTKRep()

"""

from pivsimc import SimRay, SimIntersection, SimObject, SimSurf, \
    SimCylindricalSurf, SimCircPlanarSurf, SimRectPlanarSurf, \
    SimSphericalSurf, SimRefractiveObject
import pivsimc
from numpy import *
from spivet import compat

SIM_VTKRES  = 20        # Resolution for VTK respresentations.
SIM_RAD2DEG = 180./pi


#################################################################
#
def getRayVTKRep(ray,pd):
    """
    ----
    
    ray         # SimRay object to convert.
    pd          # vtkPolyData object.
    
    ----
    
    Stores a vtkPolyLine() representation of ray in the vtkPolyData
    object, pd.
    """
    try:
        import vtk
    except:
        print "VTK support unavailable for pivsim."
        return

    pts  = pd.GetPoints()
    npts = pts.GetNumberOfPoints()
    rpts = ray.points
    
    pl = vtk.vtkPolyLine()
    pl.GetPointIds().SetNumberOfIds( len(rpts) )
    for p in range( len(rpts) ):
        pt = rpts[p]
        pts.InsertNextPoint(pt[::-1])
        pl.GetPointIds().SetId(p,p+npts)
        
    pd.InsertNextCell(pl.GetCellType(),pl.GetPointIds())

    return pd

"""
General note on lbounds.

The lbounds variable for SimSphere, etc is expressed in object local
coordinates.  The lbounds array entries are ordered similarly to the
nodes of a hex FEA element.

                    /\ c0
                    /
                   /
           o----------o  
          /.4        /|5
         / .        / |
        /  . 7     /  |
       /   o....../...o 6  
    0 o----------o   /  -------> c2 
      |  .       |1 /
      | .        | /
      |.         |/
      o----------o
       3   |     2
           |
           v c1
"""

#################################################################
#
class SimCylinder(SimRefractiveObject):
    """
    Simulation cylinder.  A cylinder's coordinate system is centered
    in the cylinder with the c0 axis pointing along the cylinder's
    logitudinal axis.  The cylinder is capped on each end.

    The object will be imageable if the refractive index is set to none.
    In this case, a ray that strikes the object will be assigned an
    intensity assuming all surfaces of the object are Lambertian.
    Otherwise, the ray will be refracted using Snell's law.
    """
    def __init__(self,pos,orient,prm,n):
        """
        ----

        pos         # Position of object origin in 3-space.
        orient      # Euler angles [phi,theta,psi] [rad].
        prm         # Geometry parameters [radius, half length].
        n           # Refractive index.

        ----

        The cylinder will have a total length of 2*prm[1].
        """
        if ( len(prm) != 2 ):
            raise ValueError, "prm must have 2 elements."

        self.m_prm    = array(prm).copy()
        self.m_lbounds = array( [[-prm[1],-prm[0],-prm[0]],
                                 [-prm[1],-prm[0], prm[0]],
                                 [-prm[1], prm[0], prm[0]],
                                 [-prm[1], prm[0],-prm[0]],
                                 [ prm[1],-prm[0],-prm[0]],
                                 [ prm[1],-prm[0], prm[0]],
                                 [ prm[1], prm[0], prm[0]],
                                 [ prm[1], prm[0],-prm[0]]])
        
        s0 = SimCylindricalSurf((0.,0.,0.),(0.,0.,0.),prm)
        s1 = SimCircPlanarSurf((prm[1],0.,0.),(0.,0.,0.),prm[0])
        s2 = SimCircPlanarSurf((-prm[1],0.,0.),(0.,pi,0.),prm[0])

        SimRefractiveObject.__init__(self,pos,orient,n,[s0,s1,s2])

    def __getattr__(self,attr):
        if ( attr == 'prm' ):
            return self.m_prm.copy()
        elif ( attr == 'lbounds' ):
            return self.m_lbounds.copy()
        else:
            return SimRefractiveObject.__getattr__(self,attr)

    def getVTKRep(self,pd,scname=None,scint=None):
        """
        ----

        pd          # vtkPolyData object.
        scname      # Scalar name.
        scint       # Scalar integer.

        ----
        
        Inserts a VTK representation of the object into the vtkPolyData
        object.  

        scname and scint are used to set a scalar variable for the object.
        SimEnv.dump2vtk() uses these arguments to set an object ID scalar
        that is unique to each object.
        """
        try:
            import vtk
        except:
            print "VTK support unavailable for pivsim."
            return None

        po = vtk.vtkCylinderSource()
        po.SetCenter((0.,0.,0.))
        po.SetResolution(SIM_VTKRES)
        po.SetRadius(self.m_prm[0])
        po.SetHeight(2.*self.m_prm[1])   # Axis is along VTK y.
        po.CappingOn()

        if ( not compat.checkNone(( not compat.checkNone(scname) ) and ( scint) ) ):
            po.GetOutput().Update()
            ncells = po.GetOutput().GetNumberOfCells()
            da = vtk.vtkIntArray()
            da.Allocate(ncells,ncells)
            da.SetName(scname)
        
            for i in range(ncells):
                da.InsertNextValue(scint)

            po.GetOutput().Update()
            po.GetOutput().GetCellData().SetScalars(da)    

        # Transform to parent coordinates.  Note: VTK uses a righthanded
        # coordinate system with positive rotations CCW.  We'll use the
        # local2parent tranform matrix.  The VTK cylinder axis is aligned
        # with parent y, so we need an extra rotation about the x-axis.
        tfrm = vtk.vtkTransform()
        tfrm.PostMultiply()
        tfrm.RotateX(90.) # Nox cylinder axis is along VTK z.

        l2p  = self.p2lrmat.transpose() 
        tmat = vtk.vtkMatrix4x4()
        tmat.Identity()
        for i in range(3):
            for j in range(3):
                tmat.SetElement(i,j,l2p[2-i,2-j])
        
            tmat.SetElement(i,3,self.pos[2-i])

        m2htfrm = vtk.vtkMatrixToLinearTransform()
        m2htfrm.SetInput(tmat) 

        tfrm.Concatenate(m2htfrm)

        tfltr = vtk.vtkTransformPolyDataFilter()
        tfltr.SetTransform(tfrm)
        tfltr.SetInput(po.GetOutput())

        # Insert the transformed object.
        apd = vtk.vtkAppendPolyData()
        apd.AddInput(pd)
        apd.AddInput(tfltr.GetOutput())

        return apd.GetOutput()


#################################################################
#
class SimRectangle(SimRefractiveObject):
    """
    Simulation Rectangle.  A rectangle's coordinate system is centered
    in the rectangle.  The prm array specifies the half-dimensions
    along the local c0, c1, c2 axes, respectively.

    The object will be imageable if the refractive index is set to none.
    In this case, a ray that strikes the object will be assigned an
    intensity assuming all surfaces of the object are Lambertian.
    Otherwise, the ray will be refracted using Snell's law.

    A SimRectangle can store a bitmap on the 0 surface (which is aligned
    with the coordinate system of the SimRectangle object and positioned 
    at prm[0] in the SimRectangle coordinates).  The surface is divided
    into the same number of pixels as the bitmap.  Any ray that strikes
    the surface will then take the interpolated intensity value given
    by the image.  All other surfaces will behave as Lambertian.

    Note: To use the bitmap feature, the refractive index must be set to
    None.
    """
    def __init__(self,pos,orient,prm,n,bitmap=None):
        """
        ----

        pos         # Position of object origin in 3-space.
        orient      # Euler angles [phi,theta,psi] [rad].
        prm         # Half-dimensions [c0,c1,c2].
        n           # Refractive index.
        bitmap      # An mxn bitmap image.

        ----

        The total dimensions of the rectangle will be 2*c0 x 2*c1 x 2*c2.
        """
        if ( len(prm) != 3 ):
            raise ValueError, "prm must have 3 elements."

        self.m_prm    = array(prm).copy()
        self.m_lbounds = array( [[-prm[0],-prm[1],-prm[2]],
                                 [-prm[0],-prm[1], prm[2]],
                                 [-prm[0], prm[1], prm[2]],
                                 [-prm[0], prm[1],-prm[2]],
                                 [ prm[0],-prm[1],-prm[2]],
                                 [ prm[0],-prm[1], prm[2]],
                                 [ prm[0], prm[1], prm[2]],
                                 [ prm[0], prm[1],-prm[2]]])

        s0 = SimRectPlanarSurf( 
            (prm[0],0.,0.), (0.,0.,0.), [prm[1],prm[2]] 
            )
        s1 = SimRectPlanarSurf( 
            (-prm[0],0.,0.), (0.,pi,0.), [prm[1],prm[2]] 
            )
        s2 = SimRectPlanarSurf( 
            (0.,prm[1],0.), (0.,-pi/2.,0.), [prm[0],prm[2]] 
            )
        s3 = SimRectPlanarSurf( 
            (0.,-prm[1],0.), (0.,pi/2.,0.), [prm[0],prm[2]] 
            )
        s4 = SimRectPlanarSurf( 
            (0.,0.,prm[2]), (pi/2.,pi/2.,0.), [prm[0],prm[1]]
            )
        s5 = SimRectPlanarSurf(
            (0.,0.,-prm[2]), (pi/2.,-pi/2.,0.), [prm[0],prm[1]]
            )

        SimRefractiveObject.__init__(self,pos,orient,n,[s0,s1,s2,s3,s4,s5])

        self.setBitmap(bitmap)

    def __getattr__(self,attr):
        if ( attr == 'prm' ):
            return self.m_prm.copy()
        elif ( attr == 'lbounds' ):
            return self.m_lbounds.copy()
        elif ( attr == 'bitmap' ):
            return self.m_surfs[0].bitmap
        else:
            return SimRefractiveObject.__getattr__(self,attr)

    def getVTKRep(self,pd,scname=None,scint=None):
        """
        ----

        pd          # vtkPolyData object.
        scname      # Scalar name.
        scint       # Scalar integer.

        ----

        Inserts a VTK representation of the object into the vtkPolyData
        object.  

        scname and scint are used to set a scalar variable for the object.
        SimEnv.dump2vtk() uses these arguments to set an object ID scalar
        that is unique to each object.
        """
        try:
            import vtk
        except:
            print "VTK support unavailable for pivsim."
            return None
        po = vtk.vtkCubeSource()
        po.SetCenter((0.,0.,0.))
        po.SetZLength(2*self.m_prm[0])
        po.SetYLength(2*self.m_prm[1])
        po.SetXLength(2*self.m_prm[2])

        if ( not compat.checkNone(( not compat.checkNone(scname) ) and ( scint) ) ):
            po.GetOutput().Update()
            ncells = po.GetOutput().GetNumberOfCells()
            da = vtk.vtkIntArray()
            da.Allocate(ncells,ncells)
            da.SetName(scname)

            for i in range(ncells):
                da.InsertNextValue(scint)

            po.GetOutput().Update()
            po.GetOutput().GetCellData().SetScalars(da)

        # Transform to parent coordinates.  Note: VTK uses a righthanded
        # coordinate system with positive rotations CCW.  We'll use the
        # local2parent tranform matrix.
        l2p  = self.p2lrmat.transpose() 
        tmat = vtk.vtkMatrix4x4()
        tmat.Identity()
        for i in range(3):
            for j in range(3):
                tmat.SetElement(i,j,l2p[2-i,2-j])
        
            tmat.SetElement(i,3,self.pos[2-i])

        m2htfrm = vtk.vtkMatrixToLinearTransform()
        m2htfrm.SetInput(tmat)

        tfltr = vtk.vtkTransformPolyDataFilter()
        tfltr.SetTransform(m2htfrm)
        tfltr.SetInput(po.GetOutput())

        # Insert the transformed object.
        apd = vtk.vtkAppendPolyData()
        apd.AddInput(pd)
        apd.AddInput(tfltr.GetOutput())

        return apd.GetOutput()

    def setBitmap(self,bitmap):
        if ( not compat.checkNone(bitmap) ):
            self.surfs[0].setBitmap(bitmap)


#################################################################
#
class SimSphere(SimRefractiveObject):
    """
    Simulation sphere.  A sphere's coordinate system is centered
    in the sphere.

    The object will be imageable if the refractive index is set to none.
    In this case, a ray that strikes the object will be assigned an
    intensity assuming all surfaces of the object are Lambertian.
    Otherwise, the ray will be refracted using Snell's law.
    """
    def __init__(self,pos,orient,radius,n):
        """
        ----

        pos         # Position of object origin in 3-space.
        orient      # Euler angles [phi,theta,psi] [rad].
        radius      # Sphere radius.
        n           # Refractive index.

        ----
        """
        self.m_radius = radius
        self.m_lbounds = array( [[-radius,-radius,-radius],
                                 [-radius,-radius, radius],
                                 [-radius, radius, radius],
                                 [-radius, radius,-radius],
                                 [ radius,-radius,-radius],
                                 [ radius,-radius, radius],
                                 [ radius, radius, radius],
                                 [ radius, radius,-radius]] )
        
        s0 = SimSphericalSurf(pos,orient,radius)
        SimRefractiveObject.__init__(self,pos,orient,n,[s0])

    def __getattr__(self,attr):
        if ( attr == 'prm' ):
            return self.m_prm.copy()
        elif ( attr == 'lbounds' ):
            return self.m_lbounds.copy()
        else:
            return SimRefractiveObject.__getattr__(self,attr)

    def getVTKRep(self,pd,scname=None,scint=None):
        """
        ----

        pd          # vtkPolyData object.
        scname      # Scalar name.
        scint       # Scalar integer.

        ----

        Inserts a VTK representation of the object into the vtkPolyData
        object.  

        scname and scint are used to set a scalar variable for the object.
        SimEnv.dump2vtk() uses these arguments to set an object ID scalar
        that is unique to each object.
        """
        try:
            import vtk
        except:
            print "VTK support unavailable for pivsim."
            return None
        po = vtk.vtkSphereSource()
        po.SetRadius(self.m_radius)
        po.SetCenter(self.pos[::-1])
        po.SetPhiResolution(SIM_VTKRES)
        po.SetThetaResolution(SIM_VTKRES)

        if ( not compat.checkNone(( not compat.checkNone(scname) ) and ( scint) ) ):
            po.GetOutput().Update()
            ncells = po.GetOutput().GetNumberOfCells()
            da = vtk.vtkIntArray()
            da.Allocate(ncells,ncells)
            da.SetName(scname)

            for i in range(ncells):
                da.InsertNextValue(scint)

            po.GetOutput().Update()
            po.GetOutput().GetCellData().SetScalars(da)

        apd = vtk.vtkAppendPolyData()
        apd.AddInput(pd)
        apd.AddInput(po.GetOutput())

        return apd.GetOutput()


#################################################################
#
class SimCamera(SimObject):
    """
    Represents a pinhole camera in 3-space.  The SimCamera object
    contains the image formed during the simulation and is also
    responsible for initializing the ray bundle.

    In terms of implementation, the 'sensor' is positioned in
    front of the pinhole.  The object local coordinate system is 
    centered on the pinhole with the c0-axis pointing toward the scene and
    penetrating the 'sensor' in the center of the sensor plane.
    The local coordinate c2-axis is oriented along the second image
    plane axis (ie, the horizontal axis in a viewed image).
    """
    def __init__(self,pos,orient,prm):
        """
        ----

        pos         # Position of object origin in 3-space.
        orient      # Euler angles [phi,theta,psi] [rad].
        prm         # Camera parameters.

        ----

        The camera is a simple pinhole model described by the following
        parameters:
            prm[0] ----- Focal length (f).
            prm[1] ----- Number of pixels along c1 (npc1).
            prm[2] ----- Physical pixel size along c1 (dpc1).
            prm[3] ----- Number of pixels along c2 (npc2).
            prm[4] ----- Physical pixel size along c2 (dpc2).

        The c0 axis is oriented such that it passes through the pinhole,
        penetrates the image plane in its center, and points toward the
        object being imaged.
        """
        SimObject.__init__(self,pos,orient)

        self.m_f      = prm[0]
        self.m_npc1   = prm[1]
        self.m_dpc1   = prm[2]
        self.m_npc2   = prm[3]
        self.m_dpc2   = prm[4]
        self.m_bitmap = zeros((prm[1],prm[3]),dtype=float)

    def __getattr__(self,attr):
        if ( attr == 'f' ):
            return self.m_f
        elif ( attr == 'npc1' ):
            return self.m_npc1
        elif ( attr == 'dpc1' ):
            return self.m_dpc1
        elif ( attr == 'npc2' ):
            return self.m_npc2
        elif ( attr == 'dpc2' ):
            return self.m_dpc2
        elif ( attr == 'prm' ):
            return [self.m_f,self.m_npc1,self.m_dpc1,self.m_npc2,self.m_dpc2]
        elif ( attr == 'bitmap' ):
            return self.m_bitmap
        else:
            return SimObject.__getattr__(self,attr)

    def initrays(self):
        """
        Creates a collection of SimRay objects, one for each pixel.  The
        rays will be created in the local coordinate system of the camera.

        Returns rays, a list of the rays.
        """
        # Initialization.
        npix    = self.m_npc1*self.m_npc2
        sources = zeros((npix,3),dtype=float)

        # Get the ray headings through pixel centers.
        pxcrds        = indices((1,self.m_npc1,self.m_npc2),dtype=float)
        pxcrds[0,...] = self.m_f
        pxcrds[1,...] = (pxcrds[1,...] -(self.m_npc1 -1.)/2.)*self.m_dpc1
        pxcrds[2,...] = (pxcrds[2,...] -(self.m_npc2 -1.)/2.)*self.m_dpc2

        pxcrds = pxcrds.reshape(3,npix).transpose()

        # Form the rays.
        rays = map(SimRay,sources,pxcrds)

        return rays


#################################################################
#
class SimLight(SimObject):
    """
    Represents a directional light of uniform intensity.  The 
    lamp will illuminate the entire environment with rays
    propagating along c0.

    The light does not cast shadows.
    """
    def __init__(self,orient):
        """
        ----
        
        orient      # Euler angles [phi,theta,psi] [rad].

        ----
        """
        SimObject.__init__(self,(0.,0.,0.),orient)
        
        liray = SimRay((0.,0.,0.),(1.,0.,0.))
        iray  = self.local2parent(liray,0)

        self.m_liray = liray
        self.m_iray  = iray

    def __getattr__(self,attr):
        if ( attr == 'liray' ):
            return self.m_liray
        elif ( attr == 'iray' ):
            return self.m_iray
        else:
            return SimObject.__getattr__(self,attr)


#################################################################
#
class SimEnv(SimRectangle):
    """
    Container for the simulation environment.  The simulation
    environment has a refractive index of n and should be large
    enough to encase all objects, including the cameras.  Its 
    primary purposes are: a) to house the other objects used in 
    the simulation, b) to provide a coordinate system from which 
    all other objects are ultimately referenced, c) to contain the
    SimOctree, and d) to propagate rays through the environment
    and form an image on the cameras.

    SimEnv objects come equipped with a default illumination source
    (ie, an instance of SimLight) that provides global, non-shadowing
    illumination.  The default light is oriented to cast rays that
    propagate along the negative x-axis (ie, c2-axis).  A different
    light can be installed using SetLight().

    The SimEnv coordinate system is like all other SimObject
    coordinate systems in that it is centered in the object.

    NOTE: Cameras are assumed to be positioned in the SimEnv
    object proper and not within another object located in the
    environment (ie, the refractive index of the material
    surrounding the cameras is that of the SimEnv).  Put another
    way, the cameras will be assumed to lie on Level 0 of the
    octree.
    """
    def __init__(self,prm,n,maxleaves=100):
        """
        ----

        prm         # Half-dimensions [c0,c1,c2].
        n           # Refractive index.
        maxleaves   # Maximum leaf nodes per internal node.

        ----

        The total dimensions of the bounding rectangle will be 
        2*c0 x 2*c1 x 2*c2.

        For more information on maxleaves, see the documentation
        for SimOctree and SimInternalNode.
        """
        SimRectangle.__init__(self,(0.,0.,0.),(0.,0.,0.),prm,n)

        self.m_cameras = []
        self.m_camrays = []
        self.m_light   = SimLight((pi/2.,-pi/2.,0.))
        self.m_octree  = SimOctree(prm,maxleaves)
        self.addObject(self)

    def __getattr__(self,attr):
        if ( attr == 'n' ):
            return self.m_n
        elif ( attr == 'cameras' ):
            return self.m_cameras
        elif ( attr == 'camrays' ):
            return self.m_camrays
        elif ( attr == 'light' ):
            return self.m_light
        elif ( attr == 'octree' ):
            return self.m_octree
        else:
            return SimRectangle.__getattr__(self,attr)

    def addObject(self,obj):
        if ( not isinstance(obj,SimRefractiveObject) ):
            raise ValueError,\
                "SimEnv can only contain objects of type SimRefractiveObject."
        
        self.m_octree.addLeaf( SimLeafNode(obj) )

    def addCamera(self,cam):
        self.m_cameras.append(cam)

    def dump2vtk(self,bofpath,mxlevel=None):
        """
        ----

        bofpath     # Base output file path.
        mxlevel     #####

        ----

        Creates three VTK output files.
           bofname-RAYS-C*.vtk ---- Rays for camera *.
           bofname-INODE.vtk ------ Octree internal nodes.
           bofname-LNODE.vtk ------ Octree leaf nodes.
        """
        # Initialization.
        try:
            import vtk
        except:
            print "VTK support unavailable for pivsim."
            return None
        dw = vtk.vtkPolyDataWriter()

        # Process rays.
        ncrc = len(self.m_camrays)
        if ( ncrc != 0 ):
            for c in range(ncrc):
                rpd = vtk.vtkPolyData()
                rpd.Allocate(1,1)
                rpd.SetPoints(vtk.vtkPoints())

                for r in self.m_camrays[c]:
                    getRayVTKRep(r,rpd)

                dw.SetInput(rpd)
                dw.SetFileName("%s-RAYS-C%i.vtk" % (bofpath,c))
                dw.SetFileTypeToASCII()
                dw.Write()        
        
        # Process octree.
        self.m_octree.dump2vtk(bofpath,mxlevel)

    def image(self):
        """
        Images the scene for all cameras.
        """
        print "STARTING: SimEnv.image"

        # Initialization.
        eps            = 1.E-6
        maxrcnt        = 5
        self.m_camrays = []
        # Raytrace.
        for c in range( len(self.m_cameras) ):
            print " | CAM %i" % c
            cam   = self.m_cameras[c]

            print " | Initializing rays."
            rays  = cam.initrays()
            dmy   = zeros(len(rays),dtype=int)
            rays  = map(cam.local2parent,rays,dmy)
            chnli = cam.bitmap
            chnli = chnli.reshape(chnli.size)
            self.m_camrays.append(rays)

            print " | Tracing ..."
            pivsimc.SimEnv_image(self,rays,chnli,maxrcnt)

        print " | EXITING: SimEnv.image"

    def refract(self,ni,nf,insc):
        """
        Refracts a ray at the intersection with surf.  Internal
        reflection can occur.

        Does not check for ni or nf = None.

        Returns a refracted ray in SimEnv coordinates.
        """
        # NOTE: This is a convenience wrapper around pivsimc.SimEnv_refract().
        # pivsimc.SimEnv_refract() is called directly from SimEnv_image().
        return pivsimc.SimEnv_refract_wrap(ni,nf,insc)

    def setLight(self,light):
        self.m_light = light


# ******************************************
# ********* OCTREE IMPLEMENTATION **********
# ******************************************

#################################################################
#
class SimNode(SimRectangle):
    """
    Base class for octree nodes.  An octree node is always aligned with
    the SimEnv coordinates.
    """
    def __init__(self,pos,prm,parent):
        """
        ----

        pos         # Position of object origin in 3-space.
        prm         # Half-dimensions [c0,c1,c2].
        parent      # Parent node.

        ----

        The total dimensions of the bounding rectangle will be 
        2*c0 x 2*c1 x 2*c2.
        """        
        SimRectangle.__init__(self,pos,(0.,0.,0.),prm,None)

        if ( ( not compat.checkNone(parent) ) and not isinstance(parent,SimNode) ):
            raise ValueError, "parent must be a valid SimNode."

        self.setParent(parent)

    def __getattr__(self,attr):
        if ( attr == 'parent' ):
            return self.m_parent
        elif ( attr == 'level' ):
            return self.m_level
        elif ( attr == 'root' ):
            return self.m_root
        else:
            return SimRectangle.__getattr__(self,attr)

    def setParent(self,parent):
        """
        ----

        parent      # Reference to the parent.

        ----

        Sets parent of the node.
        """
        self.m_parent = parent
        if ( compat.checkNone(parent) ):
            self.m_level = 0
            self.m_root  = self
        else:
            self.m_level = parent.level +1
            self.m_root  = parent.root


#################################################################
#
class SimInternalNode(SimNode):
    """
    Represents an internal node in the octree.  An internal node can have
    children that are either leaf nodes or other SimInternalNode objects,
    but not both.
    
    SimInternalNode instances are added to the octree in groups of 8.
    If the number of leaves in a given internal node exceeds maxleaves,
    then the internal node will be subdivided into 8 child internal
    nodes and the leaf nodes re-parented as necessary. 
    """
    def __init__(self,pos,prm,parent,maxleaves):
        """
        ----

        pos         # Position of object origin in octree 3-space.
        prm         # Half-dimensions [c0,c1,c2].
        parent      # Parent node.
        maxleaves   # Maximum number of leaves per node.

        ----
        If the number of leaves exceeds maxleaves, then a new group of
        8 child SimInternalNodes will be inserted.

        The total dimensions of the rectangle will be 2*c0 x 2*c1 x 2*c2.
        """ 
        SimNode.__init__(self,pos,prm,parent)
        
        self.m_maxleaves = maxleaves
        self.m_children  = []
        self.m_leaves    = []

        self.setBounds()

    def __getattr__(self,attr):
        if ( attr == 'maxleaves' ):
            return self.m_maxleaves
        elif ( attr == 'children' ):
            return self.m_children
        elif ( attr == 'leaves' ):
            return self.m_leaves
        else:
            return SimNode.__getattr__(self,attr)

    def addLeaf(self,leaf):
        """
        ----

        leaf        # Reference to the SimLeafNode object.

        ----

        Attempts to insert a leaf node into the SimInternalNode object.
        If the bounding box of the leaf node does not intersect the
        bounding box of the SimInternalNode, the leaf node is ignored.
        If, however, the leaf node does intersect the SimInternalNode
        bounding box, then the leaf node is added to the SimInternalNode
        or sent to the child nodes (if any) for consideration.
        """
        if ( not self.checkLeafBounds(leaf) ):
            # Leaf node doesn't land inside bounding box.
            return

        # If this node has no child SimInternalNode objects, then add
        # the leaf node.  Otherwise, send it to the children.
        if ( len( self.m_children ) == 0 ):
            leaf.setParent(self)
            if ( compat.checkNone(leaf.id) ):
                leaf.setID(self.root.cleafid)
                self.root.incleafid()
            self.m_leaves.append(leaf)

            if ( len ( self.m_leaves ) > self.m_maxleaves ):
                self.subdivide()                
        else:
            for child in self.m_children:
                child.addLeaf(leaf)

    def checkLeafBounds(self,leaf):
        """
        ----

        leaf        # Reference to the SimLeafNode object.

        ----

        Returns True if any part of leaf bounding box intersects the
        the bounding box of the current internal node.
        """
        # The bounding box of all internal nodes is aligned with the 
        # global coordinates, so we simply need the coordinates of the
        # min and max vertices from the bounding box.
        bounds = self.m_bounds
        mnb    = bounds.min(axis=0)
        mxb    = bounds.max(axis=0)

        # Check if any part of the leaf BB intersects the BB of this node.
        lfbounds = leaf.bounds
        mnlb     = lfbounds.min(axis=0)
        mxlb     = lfbounds.max(axis=0)
        msk      = ( mxlb > mnb ) * ( mnlb < mxb )
        return msk.all()

    def setBounds(self):
        """
        Computes the bounding box of the SimInternalNode object.
        """
        self.m_bounds = self.lbounds +self.pos

    def subdivide(self):
        """
        Called when the number of leaf nodes exceeds maxleaves.  Adds 
        eight children to the current internal node and re-parents any
        leaf nodes that exist.
        """
        cellsz  = self.m_prm
        hcellsz = cellsz/2.
        origin  = self.pos -hcellsz

        posv = indices((2,2,2),dtype=float)
        posv = posv.reshape((3,8)).transpose()
        for i in range(3):
            posv[:,i] = posv[:,i]*cellsz[i] +origin[i]

        for i in range(8):
            self.m_children.append( 
                SimInternalNode(posv[i],hcellsz,self,self.m_maxleaves) )

        # Move leaves.
        for leaf in self.m_leaves:
            leaf.removeParent(self)
            for child in self.m_children:
                child.addLeaf(leaf)

        # Zero out leaves container.
        self.m_leaves = []
            

#################################################################
#
class SimLeafNode(SimNode):
    """
    Represents a leaf node in the octree.  An leaf node can 
    hold only one SimRefractiveObject and has no children.

    A SimLeafNode can wrap very large objects that dimensionally
    span the breadth of several SimInternalNodes.  If the bounding
    box of a SimLeafNode crosses more than one SimInternalNode,
    then a reference to the SimLeafNode should be stored in each
    intruded SimInternalNode.  

    Having multiple references to a given SimLeafNode means that a 
    leaf node can have multiple parents in the tree.  Each parent
    should register with the leaf node via a call to setParent() or
    addParent().  If the leaf node is moved within the tree (due to
    subdivision of a SimInternalNode), then the previous parent should
    remove itself from the leaf node's parents dictionary by calling
    removeParent().

    The position of a SimLeafNode in the octree coordinate system
    is derived directly from the position of the contained 
    SimRefractiveObject. The SimLeafNode coordinate system, however, is
    aligned with the SimOctree coordinate system, as opposed to that of
    the wrapped SimRefractiveObject.
    """
    def __init__(self,obj):
        """
        ----

        obj         # Node object

        ----

        obj must point to a valid SimRefractiveObject.
        """ 
        self.m_obj = obj
        self.setBounds()
        
        prm = self.__getprm__(self.m_bounds)

        SimNode.__init__(self,obj.pos,prm,None)

        self.m_id = None
        self.m_parents = {}

    def __getattr__(self,attr):
        if ( attr == 'obj' ):
            return self.m_obj
        elif ( attr == 'bounds' ):
            return self.m_bounds.copy()
        elif ( attr == 'id' ):
            return self.m_id
        elif ( attr == 'parents' ):
            return self.m_parents
        elif ( attr == 'parent' ):
            return self.m_parents
        else:
            return SimNode.__getattr__(self,attr)

    def __getprm__(self,bounds):
        return bounds.max(axis=0) -self.m_obj.pos

    def addParent(self,parent):
        """
        ----

        parent      # Reference to the parent.

        ----

        Adds a parent reference to the leaf node's dictionary of parents.
        """
        self.m_parents["%X" % id(parent)] = parent
    
    def removeParent(self,parent):
        """
        ----

        parent      # Reference to the parent.

        ----

        Removes a parent reference from the leaf node's dictionary of
        parents.
        """
        self.m_parents.pop("%X" % id(parent))

    def setBounds(self):
        """
        Every SimRefractiveObject posesses an array of 8 vertices that 
        define a bounding box large enough to contain the object in the 
        object's local coordinate system.  setBounds() computes the bounding
        box vertices necessary to contain the SimRefractiveObject in the
        coordinate system of the octree.
        """
        # Convert object's bounding box coordinates into parent coordinates.
        lbounds = self.m_obj.lbounds
        dmy   = zeros(lbounds.shape[0],dtype=int)
        pbnds = array( map(self.m_obj.local2parent,lbounds,dmy) )

        # Compute a new bounding box.
        pbmn = pbnds.min(axis=0)
        pbmx = pbnds.max(axis=0)

        bounds = array( [[pbmn[0],pbmn[1],pbmn[2]],
                         [pbmn[0],pbmn[1],pbmx[2]],
                         [pbmn[0],pbmx[1],pbmx[2]],
                         [pbmn[0],pbmx[1],pbmn[2]],
                         [pbmx[0],pbmn[1],pbmn[2]],
                         [pbmx[0],pbmn[1],pbmx[2]],
                         [pbmx[0],pbmx[1],pbmx[2]],
                         [pbmx[0],pbmx[1],pbmn[2]]] )

        self.m_bounds = bounds

    def setID(self,id):
        """
        Sets the ID integer for the leaf node.  The ID is set automatically
        based on the order in which the leaf node was added to the tree.
        """
        self.m_id = id

    def setParent(self,parent):
        """
        ----

        parent      # Reference to the parent.

        ----

        Adds a parent to the parents dictionary.  setParent() simply
        calls addParent().
        """
        if ( not compat.checkNone(parent) ):
            self.addParent(parent)


#################################################################
#
class SimOctree(SimInternalNode):
    """
    Class for a scene octree.  The octree reduces the computational
    burden associated with raytracing, as testing for ray intersections
    is limited to connected nodes within the tree.  

    A SimOctree consists of a root node and a collection of additional
    nodes.  Nodes in a SimOctree consist of two types:
       SimInternalNode ----- Represents an internal node.  An internal
                             node can have child nodes that are also
                             of type SimInternalNode.  If an internal
                             node has no child SimInternalNodes,
                             then the internal node can contain up to
                             maxleaves leaf nodes (SimLeafNode).
       SimLeafNode --------- A SimLeafNode holds a single 
                             SimRefractiveObject instance.  A SimLeafNode
                             is a terminal node, as it cannot have
                             children.  Since a given SimRefractiveObject
                             can be large and span the space occupied by
                             several SimInternalNodes, a SimLeafNode can
                             be shared among many parent SimInternalNodes.

    For an overview of how octrees are employed in raytracing, consult
    Akenine-Moller:2002.
    """
    def __init__(self,prm,maxleaves):
        """
        ----

        prm         # Half-dimensions [c0,c1,c2].
        maxleaves   # Maximum number of leaves per node.

        ----
        
        The half-dimensions (prm) should be large enough to contain the 
        full scene (including cameras).

        If the SimOcree instance or any child node contained in the octree
        has more than maxleaves SimLeafNode objects, then the parent
        SimInternalNode object will be subdivided and the leaves
        re-parented..
        """
        SimInternalNode.__init__(self,(0.,0.,0.),prm,None,maxleaves)
        
        # m_cleafid is incremented every time leaf node is added to
        # the octree.  It is primarily used by dump2graphviz().  A
        # leaf node that is shared among many parent nodes will have
        # the same leaf id.
        self.m_cleafid = 0  

    def __getattr__(self,attr):
        if ( attr == 'cleafid' ):
            return self.m_cleafid
        else:
            return SimInternalNode.__getattr__(self,attr)

    def __getOctreeVTKRep__(self,ipd,lpd,mxlevel=None):
        """
        ----

        ipd         # vtkPolyData object for interal nodes.
        lpd         # vtkPolyData object for leaf nodes.
        mxlevel     # Maximum level to generate.

        ----

        Called by dump2vtk() to insert valid VTK representations of
        the internal and leaf nodes into the ipd and lpd vtkPolyData
        objects.

        NOTE:  A large simulation can generate a large tree.  mxlevel can
        be used to clip the tree to the uppermost levels.  Root corresponds
        to level 0.
        """
        # Initialization
        if ( compat.checkNone(mxlevel) ):
            mxlevel = iinfo(int).max

        queue = [self]
        inn   = 0
        lnpf  = zeros(self.root.cleafid,dtype=bool)
        while ( len(queue) > 0 ):
            node = queue.pop(0)
            if ( node.level > mxlevel ):
                break

            ipd = node.getVTKRep(ipd,"INID",inn)
            queue.extend(node.children)

            for leaf in node.leaves:
                # Check if this leaf has been already processed.
                if ( lnpf[leaf.id] ):
                    continue
                lpd           = leaf.obj.getVTKRep(lpd,"LNID",leaf.id)
                lnpf[leaf.id] = True

            inn = inn +1

        return [ipd,lpd]

    def dump2graphviz(self,ofpath,mxlevel=None):
        """
        ----

        ofpath      # Output file path.
        mxlevel     # Maximum level to draw.

        ----

        Creates a textfile at ofpath that can be processed using graphviz's
        dot or other routines.  Internal nodes, identifed by an ellipse, 
        will be numbered as they are encountered in a breadth-first search.  
        Leaf nodes, shown as rectangles, are assigned a unique integer 
        based on the order in which the leaf was added to the tree.  

        NOTE:  A large simulation can generate a large tree.  mxlevel can
        be used to clip the tree to the uppermost levels.  Root corresponds
        to level 0.
        """
        # Initialization
        if ( compat.checkNone(mxlevel) ):
            mxlevel = iinfo(int).max

        fh = open(ofpath,'w')
        
        fh.write("digraph SimOctree {\n")

        queue = [self]
        prnt  = [None]
        rnk   = []
        inn   = 0
        while ( len(queue) > 0 ):
            node = queue.pop(0)
            pin  = prnt.pop(0)
            if ( node.level > mxlevel ):
                break

            if ( not compat.checkNone(pin) ):
                fh.write("IN%i -> IN%i;\n" % (pin,inn))
            else:
                fh.write("IN%i;\n" % inn)

            queue.extend(node.children)
            
            nidl    = empty( len(node.children), dtype=int )
            nidl[:] = inn

            prnt.extend(nidl)

            if ( node.level >= len(rnk) ):
                rnk.append([])

            rnk[node.level].append(inn)

            for leaf in node.leaves:
                fh.write("LN%i [shape=box];\n" % leaf.id)
                fh.write("IN%i -> LN%i;\n" \
                             % (inn,leaf.id) )

            inn = inn +1

        for r in rnk:
            fh.write("{ rank=same;\n")
            for n in r:
                fh.write("IN%i;\n" % n)
            fh.write("}\n")

        fh.write("}\n")
        fh.close()

    def dump2vtk(self,bofpath,mxlevel=None):
        """
        ----

        bofpath     # Base output file path.
        mxlevel     # Maximum level to generate.

        ----

        dump2vtk() allows the entire simulation scene to be viewed with
        ParaView.  This is mainly useful to trouble shooting or smaller
        runs.

        Creates two VTK output files.
           bofname-INODE.vtk ------ Octree internal nodes.
           bofname-LNODE.vtk ------ Octree leaf nodes.
        """
        # Initialization.
        try:
            import vtk
        except:
            print "VTK support unavailable for pivsim."
            return None

        ipd = vtk.vtkPolyData()
        ipd.Allocate(1,1)
        ipd.SetPoints(vtk.vtkPoints())

        lpd = vtk.vtkPolyData()
        lpd.Allocate(1,1)
        lpd.SetPoints(vtk.vtkPoints())

        dw = vtk.vtkPolyDataWriter()

        # Process nodes.
        [ipd,lpd] = self.__getOctreeVTKRep__(ipd,lpd,mxlevel)
        
        dw.SetInput(ipd)
        dw.SetFileName("%s-INODE.vtk" % bofpath)
        dw.SetFileTypeToASCII()
        dw.Write()

        dw.SetInput(lpd)
        dw.SetFileName("%s-LNODE.vtk" % bofpath)
        dw.SetFileTypeToASCII()
        dw.Write()
                
    def incleafid(self):
        """
        Increments the integer used to uniquely identify SimLeafNode objects.
        incleafid() should be called every time a leaf node is inserted into
        the tree.
        """
        self.m_cleafid = self.m_cleafid +1

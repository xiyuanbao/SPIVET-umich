/*
Filename:  pivsimc.c
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
  Numeric functions for pivsim module.

*/

#include "Python.h"
#include "numpy/arrayobject.h"

#include <math.h>
#include <stdlib.h>
#include <float.h>


#define PIVSIMC_EXTSZ 10
#define PIVSIMC_DEPS 1.E-6

//
//  MACROS
//

// double a[9], x[3], b[3]
// Computes b = a*x
#define PIVSIMC_MVMUL(a,x,b) \
  if (1) { \
    b[0] = a[0]*x[0] +a[1]*x[1] +a[2]*x[2]; \
    b[1] = a[3]*x[0] +a[4]*x[1] +a[5]*x[2]; \
    b[2] = a[6]*x[0] +a[7]*x[1] +a[8]*x[2]; \
  } else (void) 0

// double a[9], x[3], b[3]
// Computes b = a'*x, where a' is the transpose.
#define PIVSIMC_MTVMUL(a,x,b) \
  if (1) { \
    b[0] = a[0]*x[0] +a[3]*x[1] +a[6]*x[2]; \
    b[1] = a[1]*x[0] +a[4]*x[1] +a[7]*x[2]; \
    b[2] = a[2]*x[0] +a[5]*x[1] +a[8]*x[2]; \
  } else (void) 0

// double a[3], b[3], c[3]
// Computes the python equivalent of c = cross(a[::-1],b[::-1])[::-1].
#define PIVSIMC_REVCROSS(a,b,c) \
  if (1) { \
    c[0] = a[2]*b[1] -a[1]*b[2]; \
    c[1] = a[0]*b[2] -a[2]*b[0]; \
    c[2] = a[1]*b[0] -a[0]*b[1]; \
  } else (void) 0

// double a[3], b[3], c
// Computes c = dot(a,b).
#define PIVSIMC_VDOTV(a,b,c) \
  if (1) { \
    c = a[0]*b[0] +a[1]*b[1] +a[2]*b[2]; \
  } else (void) 0

// double v[3], n
// Computes v = v/|v|, where n is used to store |v|.
#define PIVSIMC_NORMALIZE(v,n) \
  if (1) { \
    n = sqrt(v[0]*v[0] +v[1]*v[1] +v[2]*v[2]); \
    if ( n > PIVSIMC_DEPS ) { \
      v[0] = v[0]/n; \
      v[1] = v[1]/n; \
      v[2] = v[2]/n; \
    } \
  } else (void) 0

// int       ndim
// npy_intp *dim
// double   *data
// Copies values from data into a new numpy array and returns the result.
#define PIVSIMC_RETURN_DARRAY(ndim,dim,data) \
  if (1) { \
    PyArrayObject *ain, *aout;  \
    ain = \
      (PyArrayObject *) \
      PyArray_SimpleNewFromData(ndim, \
                                dim, \
                                PyArray_DOUBLE, \
                                data ); \
    aout = \
      (PyArrayObject *) \
      PyArray_EMPTY(ndim,dim,PyArray_DOUBLE,PyArray_CORDER); \
    PyArray_CopyInto(aout,ain); \
    cleanup_po(1,ain);           \
    return PyArray_Return(aout); \
  } else (void) 0

//~


//
// WORKER FUNCTION PROTOTYPES
//
static void cleanup_po(int nvars, ...);
//~


/////////////////////////////////////////////////////////////////
//
// class SimRay
//
// points, heads, seglengths are allocated in chunks of memory
// sufficient to store PIVSIMC_EXTSZ elements (where a point in
// 3-space is considered 1 element).
//
typedef struct {
  PyObject_HEAD
  double   *points;         // Collection of sources.
  double   *heads;          // Collection of headings.
  double   *seglengths;     // Collection of segment lengths.
  double    intensity;      // Ray intensity.
  npy_intp  pointcount;     // Number of points in points, heads.
  npy_intp  segments;       // Number of segments in seglengths.
  npy_intp  len_points;     // points.size
  npy_intp  len_seglengths; // seglengths.size
} SimRay;

static void SimRay_dealloc(SimRay *self) {
  PyMem_Free(self->points);
  PyMem_Free(self->heads);
  PyMem_Free(self->seglengths);
  self->ob_type->tp_free((PyObject *)self);
}

// head will be modified in place.
static int SimRay_init_core(SimRay *self,double *source,double *head){
  double *ptr1, *ptr2, *ptr3;
  double dmy;
  int i;

  ptr1 = (double *)PyMem_Malloc(3*PIVSIMC_EXTSZ*sizeof(double));
  ptr2 = (double *)PyMem_Malloc(3*PIVSIMC_EXTSZ*sizeof(double));
  ptr3 = (double *)PyMem_Malloc(PIVSIMC_EXTSZ*sizeof(double));
  if ( ( ptr1 == NULL ) || ( ptr2 == NULL) || ( ptr3 == NULL ) ) {
    PyMem_Free(ptr1);
    PyMem_Free(ptr2);
    PyMem_Free(ptr3);
    return -1;
  }

  self->points         = ptr1;
  self->heads          = ptr2;
  self->seglengths     = ptr3;
  self->pointcount     = 1;
  self->segments       = 0;
  self->len_points     = PIVSIMC_EXTSZ;
  self->len_seglengths = PIVSIMC_EXTSZ;
  self->intensity      = 0.;

  PIVSIMC_NORMALIZE(head,dmy);

  for ( i=0; i<3; i++ ) {
    self->points[i] = source[i];
    self->heads[i]  = head[i];
  }

  return 0;
}

static int SimRay_init(SimRay *self,PyObject *args,PyObject *kwds) {
  double source[3], head[3];

  if (!PyArg_ParseTuple(args,
                        "(ddd)(ddd)",
                        &source[0],&source[1],&source[2],
                        &head[0],&head[1],&head[2]
                        )
      )
    return -1;

  return SimRay_init_core(self,source,head);
}

// For internal calls.
static void SimRay_point_core(SimRay *self,double t,double *point) {
  int i, bndx;

  bndx = 3*(self->pointcount -1);
  for ( i = 0; i < 3; i++ )
    point[i] = self->heads[bndx +i]*t +self->points[bndx +i];
}

static PyObject *SimRay_point(SimRay *self,PyObject *args) {
  PyArrayObject *point;

  double  *ptr_point;
  double   t;
  npy_intp dim=3;

  if (!PyArg_ParseTuple(args,"d",&t))
    return NULL;

  point = (PyArrayObject *)PyArray_EMPTY(1,&dim,PyArray_DOUBLE,PyArray_CORDER);
  if ( point == NULL )
    return NULL;

  ptr_point = (double *)PyArray_DATA(point);

  SimRay_point_core(self,t,ptr_point);

  return PyArray_Return(point);
}

static PyObject *SimRay_addSegment(SimRay *self,PyObject *args) {
  double *ptr1, *ptr2, *ptr3;
  double t;
  int i, nnpts, nnsgs, sz, bndx;

  if (!PyArg_ParseTuple(args,"d",&t))
    return NULL;

  nnpts = self->pointcount +1;
  nnsgs = self->segments +1;

  if ( nnpts > self->len_points ) {
    self->len_points = self->len_points +PIVSIMC_EXTSZ;
    sz   = 3*self->len_points;
    ptr1 = (double *)PyMem_Realloc(self->points,sz*sizeof(double));
    ptr2 = (double *)PyMem_Realloc(self->heads,sz*sizeof(double));

    if ( ( ptr1 == NULL ) || ( ptr2 == NULL ) ) {
      PyMem_Free(ptr1);
      PyMem_Free(ptr2);
      return NULL;
    }

    self->points = ptr1;
    self->heads  = ptr2;
  }

  if ( nnsgs > self->len_seglengths ) {
    sz = self->len_seglengths = self->len_seglengths +PIVSIMC_EXTSZ;
    ptr3 = (double *)PyMem_Realloc(self->seglengths,sz*sizeof(double));

    if ( ptr3 == NULL )
      return NULL;

    self->seglengths = ptr3;
  }

  bndx = 3*self->pointcount;
  ptr1 = self->points +bndx;
  ptr2 = self->heads +bndx -3;  // Old head.
  ptr3 = self->heads +bndx;     // New head.

  SimRay_point_core(self,t,ptr1);
  for ( i = 0; i < 3; i++ )
    ptr3[i] = ptr2[i];

  self->seglengths[self->segments] = t;

  self->pointcount = nnpts;
  self->segments   = nnsgs;

  Py_RETURN_NONE;
}

static PyObject *SimRay_changeHeading(SimRay *self,PyObject *args) {
  double head[3], dmy;
  int i, bndx;

  if (!PyArg_ParseTuple(args,"(ddd)",&head[0],&head[1],&head[2]))
    return NULL;

  PIVSIMC_NORMALIZE(head,dmy);

  bndx = 3*(self->pointcount -1);
  for ( i = 0; i < 3; i++ )
    self->heads[bndx +i] = head[i];

  Py_RETURN_NONE;
}

static PyObject *SimRay_setIntensity(SimRay *self,PyObject *args) {
  double gsv;

  if (!PyArg_ParseTuple(args,"d",&gsv))
    return NULL;

  self->intensity = gsv;
  Py_RETURN_NONE;
}

// After call to init, we should always have a source and head.
static PyObject *SimRay_get_source(SimRay *self) {
  double *sptr;
  npy_intp dim = 3;

  sptr = self->points +3*(self->pointcount -1);
  PIVSIMC_RETURN_DARRAY(1,&dim,sptr);
}

static PyObject *SimRay_get_head(SimRay *self) {
  double *hptr;
  npy_intp dim = 3;

  hptr = self->heads +3*(self->pointcount -1);
  PIVSIMC_RETURN_DARRAY(1,&dim,hptr);
}

static PyObject *SimRay_get_intensity(SimRay *self) {
  return Py_BuildValue("d",self->intensity);
}

static PyObject *SimRay_get_pointcount(SimRay *self) {
  return Py_BuildValue("i",self->pointcount);
}

static PyObject *SimRay_get_segments(SimRay *self) {
  return Py_BuildValue("i",self->segments);
}

// points(), heads(), and seglengths() need to return a copy since
// the ray object can be deallocated independently.
static PyObject *SimRay_get_points(SimRay *self) {
  npy_intp dim[2];
  dim[0] = self->pointcount;
  dim[1] = 3;

  PIVSIMC_RETURN_DARRAY(2,dim,self->points);
}

static PyObject *SimRay_get_heads(SimRay *self) {
  npy_intp dim[2];
  dim[0] = self->pointcount;
  dim[1] = 3;

  PIVSIMC_RETURN_DARRAY(2,dim,self->heads);
}

static PyObject *SimRay_get_seglengths(SimRay *self) {
  PIVSIMC_RETURN_DARRAY(1,&self->segments,self->seglengths);
}

static PyGetSetDef SimRay_getseters[] = {
  {"source",(getter)SimRay_get_source,NULL,
   "Returns current source as numpy array.",NULL},
  {"head",(getter)SimRay_get_head,NULL,
   "Returns current heading as numpy array.",NULL},
  {"intensity",(getter)SimRay_get_intensity,NULL,
   "Returns current intensity.",NULL},
  {"pointcount",(getter)SimRay_get_pointcount,NULL,
   "Returns the current number of ray points."},
  {"segments",(getter)SimRay_get_segments,NULL,
   "Returns the current number of ray segments.",NULL},
  {"points",(getter)SimRay_get_points,NULL,
   "Returns a numpy array containing all of the ray points.",NULL},
  {"heads",(getter)SimRay_get_heads,NULL,
   "Returns a numpy array containing all of the ray headings.",NULL},
  {"seglengths",(getter)SimRay_get_seglengths,NULL,
   "Returns a numpy array containing all of the ray segment lengths.",NULL},
  {NULL}
};


static PyMethodDef SimRay_methods[] = {
  {"point",(PyCFunction)SimRay_point,METH_VARARGS,
   "SimRay_point(t)\n"
   "----\n\n"
   "t           # Distance along ray in direction of heading.\n\n"
   "----\n\n"
   "Computes coordinates of point a distance t along ray from\n"
   "current source point.\n"
   "\n"
   "Returns a 3-element array of point coordinates expressed\n"
   "as (c0,c1,c2) in the local coordinate system under which the ray\n"
   "is expressed.\n"
  },
  {"addSegment",(PyCFunction)SimRay_addSegment,METH_VARARGS,
   "SimRay_addSegment(t)\n"
   "----\n\n"
   "t           # Distance along ray in direction of heading.\n\n"
   "----\n\n"
   "Adds a new segment to the ray.\n"
   "\n"
   "Each new segment is assumed to start at the end of the\n"
   "previous segment."
  },
  {"changeHeading",(PyCFunction)SimRay_changeHeading,METH_VARARGS,
   "SimRay_changeHeading(head)\n"
   "----\n\n"
   "head        # New heading (c0,c1,c2).\n\n"
   "----\n\n"
   "Updates current heading."
  },
  {"setIntensity",(PyCFunction)SimRay_setIntensity,METH_VARARGS,
   "SimRay_setIntensity(intensity)\n"
   "----\n\n"
   "intensity   # Intensity value.\n\n"
   "----\n\n"
   "Sets intensity."
  },
  {NULL}
};

static PyTypeObject SimRayType = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "pivsimc.SimRay",          /*tp_name*/
    sizeof(SimRay),            /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)SimRay_dealloc, /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    0,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    "A simple ray is merely a vector that originates at some\n"
    "specified point (ie, it has both an origin and a heading).\n"
    "\n"
    "The SimRay class extends the concept of a simple ray by\n"
    "recording the path the ray has taken as it propagates through\n"
    "space.  That is a SimRay can have kinks due to interaction with\n"
    "refractive objects.\n"
    "\n"
    "To create a SimRay object:\n"
    "    __init__(source,head)\n"
    "    ----\n\n"
    "    source      # Ray starting point.\n"
    "    head        # Heading vector.\n\n"
    "    ----\n\n"
    "    NOTE: head will be normalized upon object instantiation.\n",
    /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    SimRay_methods,            /* tp_methods */
    0,                         /* tp_members */
    SimRay_getseters,          /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)SimRay_init,     /* tp_init */
    0,                         /* tp_alloc */
    PyType_GenericNew,         /* tp_new */
};


/////////////////////////////////////////////////////////////////
//
// class SimIntersection
//
typedef struct {
  PyObject_HEAD
  PyObject *obj;
  int       sndx;
  SimRay   *lray;
  SimRay   *sray;
  double    t;
  double    lnml[3];
  double    snml[3];
  int       exflg;
  int       tir;       // Total internal reflection flag.
} SimIntersection;

static void SimIntersection_dealloc(SimIntersection *self) {
  cleanup_po(3,self->obj,self->lray,self->sray);
  self->ob_type->tp_free((PyObject *)self);
}

static int
SimIntersection_init(SimIntersection *self,PyObject *args, PyObject *kwds) {
  if (!PyArg_ParseTuple(args,
                        "OiOOd(ddd)(ddd)i",
                        &self->obj,&self->sndx,&self->lray,&self->sray,
                        &self->t,&self->lnml[0],&self->lnml[1],&self->lnml[2],
                        &self->snml[0],&self->snml[1],&self->snml[2],
                        &self->exflg ) )
    return -1;

  Py_INCREF(self->obj);
  Py_INCREF(self->lray);
  Py_INCREF(self->sray);

  self->tir = 0;

  return 0;
}

static PyObject *SimIntersection_get_obj(SimIntersection *self) {
  Py_INCREF(self->obj);
  return self->obj;
}

static PyObject *SimIntersection_get_sndx(SimIntersection *self) {
  return Py_BuildValue("i",self->sndx);
}

static PyObject *SimIntersection_get_lray(SimIntersection *self) {
  Py_INCREF(self->lray);
  return (PyObject *)self->lray;
}

static PyObject *SimIntersection_get_sray(SimIntersection *self) {
  Py_INCREF(self->sray);
  return (PyObject *)self->sray;
}

static PyObject *SimIntersection_get_t(SimIntersection *self) {
  return Py_BuildValue("d",self->t);
}

static PyObject *SimIntersection_get_lnml(SimIntersection *self) {
  npy_intp dim = 3;
  PIVSIMC_RETURN_DARRAY(1,&dim,self->lnml);
}

static PyObject *SimIntersection_get_snml(SimIntersection *self) {
  npy_intp dim = 3;
  PIVSIMC_RETURN_DARRAY(1,&dim,self->snml);
}

static PyObject *SimIntersection_get_exflg(SimIntersection *self) {
  return PyBool_FromLong(self->exflg);
}

static PyGetSetDef SimIntersection_getseters[] = {
  {"obj",(getter)SimIntersection_get_obj,NULL,"Intersected object.",NULL},
  {"sndx",(getter)SimIntersection_get_sndx,NULL,"Surface index.",NULL},
  {"lray",(getter)SimIntersection_get_lray,NULL,
   "Ray in object local coords.",NULL},
  {"sray",(getter)SimIntersection_get_sray,NULL,
   "Ray in surface coords.",NULL},
  {"t",(getter)SimIntersection_get_t,NULL,
   "Distance along ray to intersection.",NULL},
  {"lnml",(getter)SimIntersection_get_lnml,NULL,
   "Surface normal in object local coordinates at intersection.",NULL},
  {"snml",(getter)SimIntersection_get_snml,NULL,
   "Surface normal in surface coordinates at intersection.",NULL},
  {"exflg",(getter)SimIntersection_get_exflg,NULL,"Exiting flag.",NULL},
  {NULL}
};

static PyTypeObject SimIntersectionType = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "pivsimc.SimIntersection", /*tp_name*/
    sizeof(SimIntersection),   /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)SimIntersection_dealloc, /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    0,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    "Simple class to track SimRefractiveObject intersection information.\n"
    "When a SimRay is intersected with a SimRefractiveObject, a\n"
    "SimIntersection object is returned."
    "\n"
    "To create a SimIntersection object:\n"
    "    __init__(obj,sndx,lray,sray,t,lnml,snml,exflg)\n"
    "    ----\n\n"
    "    obj         # Reference to the object intersected.\n"
    "    sndx        # Surface index.\n"
    "    lray        # Ray in object local (not surface local) coords.\n"
    "    sray        # Ray in surface local coords.\n"
    "    t           # Distance along ray to intersection.\n"
    "    lnml        # Surface normal at intersection in object local coords.\n"
    "    snml        # Surface normal at intersection is surface local\n"
    "                # coords.\n"
    "    exflg       # Exiting flag.\n\n"
    "    ----\n\n"
    "    If exflg is True, then the ray is leaving the object at the\n"
    "    point of intersection.\n", /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    0,                         /* tp_methods */
    0,                         /* tp_members */
    SimIntersection_getseters, /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)SimIntersection_init,  /* tp_init */
    0,                         /* tp_alloc */
    PyType_GenericNew,         /* tp_new */
};


/////////////////////////////////////////////////////////////////
//
// class SimObject
//
typedef struct {
  PyObject_HEAD
  double   pos[3];
  double   orient[3];
  double   p2lrmat[9];
} SimObject;

static void SimObject_dealloc(SimObject *self) {
  self->ob_type->tp_free((PyObject *)self);
}

static void SimObject_buildrmat(SimObject *self) {
  double *orient, *p2lrmat;
  double cphi, sphi, ctheta, stheta, cpsi, spsi;

  orient  = self->orient;
  p2lrmat = self->p2lrmat;

  cphi   = cos(orient[0]);
  sphi   = sin(orient[0]);
  ctheta = cos(orient[1]);
  stheta = sin(orient[1]);
  cpsi   = cos(orient[2]);
  spsi   = sin(orient[2]);

  p2lrmat[0] =  ctheta;
  p2lrmat[1] = -stheta*cphi;
  p2lrmat[2] =  stheta*sphi;
  p2lrmat[3] =  cpsi*stheta;
  p2lrmat[4] = -spsi*sphi +ctheta*cphi*cpsi;
  p2lrmat[5] = -spsi*cphi -ctheta*sphi*cpsi;
  p2lrmat[6] =  spsi*stheta;
  p2lrmat[7] =  cpsi*sphi +ctheta*cphi*spsi;
  p2lrmat[8] =  cpsi*cphi -ctheta*sphi*spsi;
}

static int SimObject_init(SimObject *self,PyObject *args,PyObject *kwds) {
  double *pos, *orient;

  pos    = self->pos;
  orient = self->orient;
  if (!PyArg_ParseTuple(args,
                        "(ddd)(ddd)",
                        &pos[0],&pos[1],&pos[2],
                        &orient[0],&orient[1],&orient[2]
                        )
      )
    return -1;

  SimObject_buildrmat(self);
  return 0;
}

static PyObject *SimObject_setpos(SimObject *self,PyObject *args) {
  double *pos;

  pos = self->pos;
  if (!PyArg_ParseTuple(args,"(ddd)",&pos[0],&pos[1],&pos[2]))
    return NULL;

  Py_RETURN_NONE;
}

static PyObject *SimObject_setorient(SimObject *self,PyObject *args) {
  double *orient;

  orient = self->orient;
  if (!PyArg_ParseTuple(args,
                        "(ddd)",
                        &orient[0],&orient[1],&orient[2]
                        )
      )
    return NULL;

  SimObject_buildrmat(self);
  Py_RETURN_NONE;
}

static PyObject *SimObject_parent2local(SimObject *self,PyObject *args) {
  PyObject      *geo;
  SimRay        *ray;
  PyArrayObject *shobj=NULL;
  double        *dptr;

  double source[3], head[3], ardata[3];
  int    i, isvec;

  npy_intp dim=3;

  if (!PyArg_ParseTuple(args,"Oi",&geo,&isvec))
    return NULL;

  if ( PyObject_TypeCheck(geo,&SimRayType) ) {
    ray  = (SimRay *)geo;

    dptr = ray->points +3*(ray->pointcount -1);
    for ( i = 0; i < 3; i++ )
      ardata[i] = dptr[i] -self->pos[i];
    PIVSIMC_MVMUL(self->p2lrmat,ardata,source);

    dptr = ray->heads +3*(ray->pointcount -1);
    PIVSIMC_MVMUL(self->p2lrmat,dptr,head);

    geo = (PyObject *)PyObject_New(SimRay,&SimRayType);
    if ( SimRay_init_core((SimRay *)geo,source,head) ) {
      cleanup_po(1,geo);
      return NULL;
    }

    return geo;
  }
  else if (isvec) {
    shobj = (PyArrayObject *)PyArray_ContiguousFromAny(geo,PyArray_DOUBLE,1,1);
    dptr  = (double *)PyArray_DATA(shobj);

    PIVSIMC_MVMUL(self->p2lrmat,dptr,head);
  }
  else {
    shobj = (PyArrayObject *)PyArray_ContiguousFromAny(geo,PyArray_DOUBLE,1,1);
    dptr  = (double *)PyArray_DATA(shobj);
    for ( i = 0; i < 3; i++ )
      source[i] = dptr[i] -self->pos[i];

    PIVSIMC_MVMUL(self->p2lrmat,source,head);
  }

  cleanup_po(1,shobj);
  PIVSIMC_RETURN_DARRAY(1,&dim,head);
}

static PyObject *SimObject_local2parent(SimObject *self,PyObject *args) {
  PyObject      *geo;
  SimRay        *ray;
  PyArrayObject *shobj;
  double        *dptr;

  double source[3], head[3];
  int    i, isvec;

  npy_intp dim=3;

  if (!PyArg_ParseTuple(args,"Oi",&geo,&isvec))
    return NULL;

  if ( PyObject_TypeCheck(geo,&SimRayType) ) {
    ray = (SimRay *)geo;

    dptr = ray->points +3*(ray->pointcount -1);
    PIVSIMC_MTVMUL(self->p2lrmat,dptr,source);
    for ( i = 0; i < 3; i++ )
      source[i] = source[i] +self->pos[i];

    dptr = ray->heads +3*(ray->pointcount -1);
    PIVSIMC_MTVMUL(self->p2lrmat,dptr,head);

    geo = (PyObject *)PyObject_New(SimRay,&SimRayType);
    if ( SimRay_init_core((SimRay *)geo,source,head) ) {
      cleanup_po(1,geo);
      return NULL;
    }

    return geo;
  }
  else if (isvec) {
    shobj = (PyArrayObject *)PyArray_ContiguousFromAny(geo,PyArray_DOUBLE,1,1);
    dptr  = (double *)PyArray_DATA(shobj);

    PIVSIMC_MTVMUL(self->p2lrmat,dptr,head);
  }
  else {
    shobj = (PyArrayObject *)PyArray_ContiguousFromAny(geo,PyArray_DOUBLE,1,1);
    dptr  = (double *)PyArray_DATA(shobj);

    PIVSIMC_MTVMUL(self->p2lrmat,dptr,head);

    for ( i = 0; i < 3; i++ )
      head[i] = head[i] +self->pos[i];
  }
  cleanup_po(1,shobj);
  PIVSIMC_RETURN_DARRAY(1,&dim,head);
}

static PyObject *SimObject_get_pos(SimObject *self) {
  npy_intp dim = 3;
  PIVSIMC_RETURN_DARRAY(1,&dim,self->pos);
}

static PyObject *SimObject_get_orient(SimObject *self) {
  npy_intp dim = 3;
  PIVSIMC_RETURN_DARRAY(1,&dim,self->orient);
}

static PyObject *SimObject_get_p2lrmat(SimObject *self) {
  npy_intp dim[2] = {3,3};
  PIVSIMC_RETURN_DARRAY(2,dim,self->p2lrmat);
}

static PyGetSetDef SimObject_getseters[] = {
  {"pos",(getter)SimObject_get_pos,NULL,
   "Returns object position as numpy array.",NULL},
  {"orient",(getter)SimObject_get_orient,NULL,
   "Returns object orientation as numpy array.",NULL},
  {"p2lrmat",(getter)SimObject_get_p2lrmat,NULL,
   "Returns the parent2local rotation matrix as a 3x3 numpy array.",NULL},
  {NULL}
};

static PyMethodDef SimObject_methods[] = {
  {"setpos",(PyCFunction)SimObject_setpos,METH_VARARGS,
   "setpos(pos,orient)\n"
   "----\n\n"
   "pos         # Position of object origin in 3-space.\n"
   "orient      # Euler angles [phi,theta,psi] [rad].\n\n"
   "----\n\n"
   "A SimObject must be initialized with a position (pos) and\n"
   "orientation (orient).  Let the parent coordinate system be\n"
   "spanned by (p0,p1,p2) which would correspond to (z,y,x) for\n"
   "the outermost object (ie, SimEnv).  pos must be ordered as\n"
   "(p0,p1,p2).  orient must be ordered as the rotation angle\n"
   "about c0 (ie, p0), then the new c2, then the new new c0.\n"
  },
  {"setorient",(PyCFunction)SimObject_setorient,METH_VARARGS,
   "setorient(orient)\n"
   "----\n\n"
   "orient      # Euler angles [phi,theta,psi].\n\n"
   "----\n\n"
   "Resets the object orientation.  See __init__() for more details."
  },
  {"parent2local",(PyCFunction)SimObject_parent2local,METH_VARARGS,
   "parent2local(geo,isvec)\n"
   "----\n\n"
   "geo         # Geometry primitive to be converted.\n"
   "isvec       # Vector flag.\n\n"
   "----\n\n"
   "Rotates from the parent coordinate system to the local\n"
   "coordinate system.\n"
   "\n"
   "geo can be one of three basic entities:\n"
   "    point ----- geo is an array with three elements describing\n"
   "                a point.  isvec must be False.\n"
   "    vector ---- geo is an array with three elements describing\n"
   "                a vector.  isvec = True.\n"
   "    SimRay ---- geo is a SimRay object.  isvec is ignored.\n"
   "\n"
   "Returns a converted entity of the same type as geo.\n"
  },
  {"local2parent",(PyCFunction)SimObject_local2parent,METH_VARARGS,
   "local2parent(geo,isvec)\n"
   "----\n\n"
   "geo         # Geometry primitive to be converted.\n"
   "isvec       # Vector flag.\n\n"
   "----\n\n"
   "Rotates from the local coordinate system to the parent\n"
   "coordinate system.\n"
   "\n"
   "geo can be one of three basic entities:\n"
   "    point ----- geo is an array with three elements describing\n"
   "                a point.  isvec must be False.\n"
   "    vector ---- geo is an array with three elements describing\n"
   "                a vector.  isvec = True.\n"
   "    SimRay ---- geo is a SimRay object.  isvec is ignored.\n"
   "\n"
   "Returns a converted entity of the same type as geo.\n"
  },
  {NULL}
};

static PyTypeObject SimObjectType = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "pivsimc.SimObject",       /*tp_name*/
    sizeof(SimObject),         /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)SimObject_dealloc, /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    0,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    "Base class for all simulation objects.  A SimObject provides\n"
    "facilities to track object location in the parent coordinate\n"
    "system as well as mechanisms to move between the parent and\n"
    "an object-specific coordinate system.\n"
    "\n"
    "To create a SimObject object:\n"
    "    __init__(pos,orient)\n"
    "    ----\n\n"
    "    pos         # Position of object origin in 3-space.\n"
    "    orient      # Euler angles [phi,theta,psi].\n\n"
    "    ----\n\n"
    "    A SimObject must be initialized with a position (pos) and\n"
    "    orientation (orient).  Let the parent coordinate system be\n"
    "    spanned by (p0,p1,p2) which would correspond to (z,y,x) for\n"
    "    the outermost object (ie, SimEnv).  pos must be ordered as\n"
    "    (p0,p1,p2).  orient must be ordered as the rotation angle\n"
    "    about c0 (ie, p0), then the new c2, then the new new c0.\n",
    /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    SimObject_methods,         /* tp_methods */
    0,                         /* tp_members */
    SimObject_getseters,       /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)SimObject_init,  /* tp_init */
    0,                         /* tp_alloc */
    PyType_GenericNew,         /* tp_new */
};


/////////////////////////////////////////////////////////////////
//
// class SimSurf
//
typedef struct {
  PyObject_HEAD
  SimObject bSimObject;
} SimSurf;

static void SimSurf_dealloc(SimSurf *self) {
  self->ob_type->tp_free((PyObject *)self);
}

static int SimSurf_init(SimSurf *self,PyObject *args,PyObject *kwds) {

  if ( SimObjectType.tp_init((PyObject *)self,args,kwds) )
    return -1;

  return 0;
}

static PyObject *SimSurf_intensity(SimSurf *self,PyObject *args) {
  SimIntersection *insc;
  SimRay *liray;
  double *snml, *head;
  double intensity;

  if (!PyArg_ParseTuple(args,"OO",&insc,&liray))
    return NULL;

  head = liray->heads +3*(liray->pointcount -1);
  snml = insc->snml;

  PIVSIMC_VDOTV(head,snml,intensity);
  intensity = fabs( fmin(0.,intensity) );

  return Py_BuildValue("d",intensity);
}

static PyMethodDef SimSurf_methods[] = {
  {"intensity",(PyCFunction)SimSurf_intensity,METH_VARARGS,
   "intensity(insc,liray)\n"
   "----\n\n"
   "insc         # SimIntersection object for camera ray.\n"
   "liray        # Illumination ray in surface local coordinates\n"
   "               (c0,c1,c2).\n\n"
   "----\n\n"
   "Computes the intensity assuming the surface is Lambertian.\n"
   "\n"
   "Returns ntns, where ntns = cos(beta), and beta\n"
   "is the angle between the surface normal and liray.\n"
  },
  {NULL}
};

static PyTypeObject SimSurfType = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "pivsimc.SimSurf",         /*tp_name*/
    sizeof(SimSurf),           /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)SimSurf_dealloc, /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    0,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    "Base class for surfaces.\n\n"
    "\n"
    "To create a SimSurf object:\n"
    "    __init__(pos,orient)\n"
    "    ----\n\n"
    "    pos         # Position of object origin in 3-space.\n"
    "    orient      # Euler angles [phi,theta,psi] [rad].\n\n"
    "    ----\n\n",            /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    SimSurf_methods,           /* tp_methods */
    0,                         /* tp_members */
    0,                         /* tp_getset */
    &SimObjectType,            /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)SimSurf_init,    /* tp_init */
    0,                         /* tp_alloc */
    PyType_GenericNew,         /* tp_new */
};


/////////////////////////////////////////////////////////////////
//
// class SimCylindricalSurf
//
typedef struct {
  PyObject_HEAD
  SimSurf   bSimSurf;
  double    radius;
  double    rsq;
  double    halflength;
} SimCylindricalSurf;

static void SimCylindricalSurf_dealloc(SimCylindricalSurf *self) {
  self->ob_type->tp_free((PyObject *)self);
}

static int
SimCylindricalSurf_init(SimCylindricalSurf *self,
                        PyObject *args,PyObject *kwds) {
  PyObject *pos, *orient, *dargs;

  if (!PyArg_ParseTuple(args,"OO(dd)",
                        &pos,&orient,&self->radius,&self->halflength))
    return -1;

  self->rsq = self->radius*self->radius;

  dargs = Py_BuildValue("OO",pos,orient);
  if ( SimSurfType.tp_init((PyObject *)self,dargs,kwds) ) {
    cleanup_po(1,dargs);
    return -1;
  }

  cleanup_po(1,dargs);

  return 0;
}

static PyObject *
SimCylindricalSurf_get_radius(SimCylindricalSurf *self) {
  return Py_BuildValue("d",self->radius);
}

static PyObject *
SimCylindricalSurf_get_halflength(SimCylindricalSurf *self) {
  return Py_BuildValue("d",self->halflength);
}

static PyObject *
SimCylindricalSurf_get_prm(SimCylindricalSurf *self) {
  return Py_BuildValue("dd",self->radius,self->halflength);
}

// The following method loosely follows that of Akenine-Moller-2002,
// Section 13.5.2.
static PyObject *
SimCylindricalSurf_intersect(SimCylindricalSurf *self,PyObject *args) {
  PyObject *otv, *dargs;
  SimRay   *lray;
  double   *lsource, *lhead, *plv;

  double phv[2], pls, phs, s, ms, q, ptv[2], ic, t;
  int    ptc, i, hiflg;

  if (!PyArg_ParseTuple(args,"O",&lray))
    return NULL;

  ptc     = 3*(lray->pointcount -1);
  lsource = lray->points +ptc;
  lhead   = lray->heads +ptc;

  // Check for any intersection by projecting onto c1-c2 plane.
  plv = lsource +1;
  pls = plv[0]*plv[0] +plv[1]*plv[1];

  phv[0] = lhead[1];
  phv[1] = lhead[2];
  phs    = phv[0]*phv[0] +phv[1]*phv[1];

  if ( phs < PIVSIMC_DEPS ) {
    // Ray is parallel to cylindrical axis.
    Py_RETURN_NONE;
  }

  phv[0] = phv[0]/sqrt(phs);
  phv[1] = phv[1]/sqrt(phs);

  s = plv[0]*phv[0] +plv[1]*phv[1];

  if ( ( s > 0. ) && ( pls > self->rsq ) ) {
    // Cylinder is behind ray source and ray is propagating away.
    Py_RETURN_NONE;
  }

  ms = pls -s*s;

  if ( ms > self->rsq ) {
    //Ray overshot cylinder.
    Py_RETURN_NONE;
  }

  // We have an intersection in the plane.
  q      = sqrt(self->rsq -ms);
  ptv[1] = -1.;
  if ( pls > self->rsq ) {
    ptv[0] = -(s +q);
    if ( q > PIVSIMC_DEPS )
      ptv[1] = -(s -q);
  }
  else
    ptv[0] = q -s;

  otv   = PyList_New(0);
  hiflg = 0;
  for ( i = 0; i < 2; i++ ) {
    if ( ptv[i] < PIVSIMC_DEPS )
      continue;

    // Get planar coordinate of intersection and solve for the 3D t.
    if ( fabs(lhead[1]) < PIVSIMC_DEPS ) {
      ic = phv[1]*ptv[i] +plv[1];
      t  = (ic -plv[1])/lhead[2];
    }
    else {
      ic = phv[0]*ptv[i] +plv[0];
      t  = (ic -plv[0])/lhead[1];
    }

    // Check that c0 coordinate of intersection is withing bounds.
    ic = lhead[0]*t +lsource[0];
    if ( fabs(ic) > self->halflength )
      continue;

    hiflg = 1;
    dargs = PyFloat_FromDouble(t);
    PyList_Append(otv,dargs);
    Py_DECREF(dargs);
  }

  if ( hiflg )
    return otv;

  cleanup_po(1,otv);
  Py_RETURN_NONE;
}

static PyObject *
SimCylindricalSurf_normal(SimCylindricalSurf *self,PyObject *args) {
  PyArrayObject *nrml;
  double *ptr_nrml;
  double lpnt[3];
  npy_intp dim = 3;

  if (!PyArg_ParseTuple(args,"(ddd)",&lpnt[0],&lpnt[1],&lpnt[2]))
    return NULL;

  nrml = (PyArrayObject *)PyArray_EMPTY(1,&dim,PyArray_DOUBLE,PyArray_CORDER);
  if (nrml == NULL )
    return NULL;

  ptr_nrml = (double *)PyArray_DATA(nrml);

  ptr_nrml[0] = 0.;
  ptr_nrml[1] = lpnt[1]/self->radius;
  ptr_nrml[2] = lpnt[2]/self->radius;

  return PyArray_Return(nrml);
}

static PyGetSetDef SimCylindricalSurf_getseters[] = {
  {"radius",(getter)SimCylindricalSurf_get_radius,NULL,
   "Returns radius.",NULL},
  {"halflength",(getter)SimCylindricalSurf_get_halflength,NULL,
   "Returns halflength.",NULL},
  {"prm",(getter)SimCylindricalSurf_get_prm,NULL,
   "Returns [radius,halflength].",NULL},
  {NULL},
};

static PyMethodDef SimCylindricalSurf_methods[] = {
  {"intersect",(PyCFunction)SimCylindricalSurf_intersect,METH_VARARGS,
   "intersect(lray)\n"
   "----\n\n"
   "lray        # Ray in local coordinates (c0,c1,c2).\n\n"
   "----\n\n"
   "A ray can intersect a cylidrical surface twice, so a list\n"
   "of all intersection distances along lray.head is returned.  The \n"
   "list is ordered with the closest intersection listed first.  If only\n"
   "one intersection is found, then a list containing one entry will be\n"
   "returned.  If no intersections are found, returns None.\n"
  },
  {"normal",(PyCFunction)SimCylindricalSurf_normal,METH_VARARGS,
   "normal(lpnt)\n"
   "----\n\n"
   "lpnt    # Point in local coordinate system (c0,c1,c2).\n\n"
   "----\n\n"
   "Computes the unit surface normal at the local surface point\n"
   "pnt (c0,c1,c2).\n"
   "\n"
   "NOTE:  lpnt should be on the surface as the computed normal\n"
   "vector is normalized by the radius.\n"
  },
  {NULL},
};

static PyTypeObject SimCylindricalSurfType = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "pivsimc.SimCylindricalSurf", /*tp_name*/
    sizeof(SimCylindricalSurf), /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)SimCylindricalSurf_dealloc, /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    0,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    "The cylindrical surface is centered around a local coordinate system\n"
    "with the c0 axis pointing along the longitudinal axis of the cylinder.\n"
    "This surface is essentially a tube with no end-caps.\n"
    "\n"
    "To create a SimCylindricalSurf object:\n"
    "    __init__(pos,orient,prm)\n"
    "    ----\n\n"
    "    pos         # Position of object origin in 3-space.\n"
    "    orient      # Euler angles [phi,theta,psi] [rad].\n"
    "    prm         # Geometry parameters [radius, half length].\n\n"
    "    ----\n\n"
    "    The cylinder will have a total length of 2*prm[1].\n", /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    SimCylindricalSurf_methods,  /* tp_methods */
    0,                         /* tp_members */
    SimCylindricalSurf_getseters, /* tp_getset */
    &SimSurfType,              /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)SimCylindricalSurf_init, /* tp_init */
    0,                         /* tp_alloc */
    PyType_GenericNew,         /* tp_new */
};


/////////////////////////////////////////////////////////////////
//
// class SimCircPlanarSurf
//
typedef struct {
  PyObject_HEAD
  SimSurf   bSimSurf;
  double    radius;
  double    rsq;
} SimCircPlanarSurf;

static void SimCircPlanarSurf_dealloc(SimCircPlanarSurf *self) {
  self->ob_type->tp_free((PyObject *)self);
}

static int
SimCircPlanarSurf_init(SimCircPlanarSurf *self,
                        PyObject *args,PyObject *kwds) {
  PyObject *pos, *orient, *dargs;

  if (!PyArg_ParseTuple(args,"OOd",
                        &pos,&orient,&self->radius))
    return -1;

  self->rsq = self->radius*self->radius;

  dargs = Py_BuildValue("OO",pos,orient);
  if ( SimSurfType.tp_init((PyObject *)self,dargs,kwds) ) {
    cleanup_po(1,dargs);
    return -1;
  }

  cleanup_po(1,dargs);

  return 0;
}

static PyObject *
SimCircPlanarSurf_get_radius(SimCircPlanarSurf *self) {
  return Py_BuildValue("d",self->radius);
}

static PyObject *
SimCircPlanarSurf_intersect(SimCircPlanarSurf *self,PyObject *args) {
  SimRay   *lray;
  double   *lsource, *lhead;

  double ic[2], rsq, t;
  int    ptc;

  if (!PyArg_ParseTuple(args,"O",&lray))
    return NULL;

  ptc     = 3*(lray->pointcount -1);
  lsource = lray->points +ptc;
  lhead   = lray->heads +ptc;

  // Check for intersections.
  if ( fabs(lhead[0]) < PIVSIMC_DEPS ) {
    // Ray is parallel to the surface.
    Py_RETURN_NONE;
  }

  t = -lsource[0]/lhead[0];
  if ( t < 0 ) {
    // Surface is behind source and ray propagating away.
    Py_RETURN_NONE;
  }

  ic[0] = lhead[1]*t +lsource[1];
  ic[1] = lhead[2]*t +lsource[2];

  rsq = ic[0]*ic[0] +ic[1]*ic[1];

  if ( rsq > self->rsq ) {
    Py_RETURN_NONE;
  }

  return Py_BuildValue("[d]",t);
}

static PyObject *
SimCircPlanarSurf_normal(SimCircPlanarSurf *self,PyObject *args) {
  PyArrayObject *nrml;
  double *ptr_nrml;
  npy_intp dim = 3;

  nrml = (PyArrayObject *)PyArray_EMPTY(1,&dim,PyArray_DOUBLE,PyArray_CORDER);
  if (nrml == NULL )
    return NULL;

  ptr_nrml = (double *)PyArray_DATA(nrml);

  ptr_nrml[0] = 1.;
  ptr_nrml[1] = 0.;
  ptr_nrml[2] = 0.;

  return PyArray_Return(nrml);
}

static PyGetSetDef SimCircPlanarSurf_getseters[] = {
  {"radius",(getter)SimCircPlanarSurf_get_radius,NULL,
   "Returns radius.",NULL},
  {NULL},
};

static PyMethodDef SimCircPlanarSurf_methods[] = {
  {"intersect",(PyCFunction)SimCircPlanarSurf_intersect,METH_VARARGS,
   "intersect(lray)\n"
   "----\n\n"
   "lray        # Ray in local coordinates (c0,c1,c2).\n\n"
   "----\n\n"
   "Returns a list containing the distance along lray.head from lray.source\n"
   "to the single surface intersection, if any. Returns None otherwise.\n"
  },
  {"normal",(PyCFunction)SimCircPlanarSurf_normal,METH_VARARGS,
   "normal(lpnt)\n"
   "----\n\n"
   "lpnt    # Point in local coordinate system (c0,c1,c2).\n\n"
   "----\n\n"
   "Computes the unit surface normal at the local surface point\n"
   "pnt (c0,c1,c2).\n"
   "\n"
   "For this surface, the returned normal vector will always\n"
   "point along the surface local c0-axis regardless of lpnt.\n"
  },
  {NULL},
};

static PyTypeObject SimCircPlanarSurfType = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "pivsimc.SimCircPlanarSurf", /*tp_name*/
    sizeof(SimCircPlanarSurf), /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)SimCircPlanarSurf_dealloc, /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    0,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    "The planar surface is centered around a local coordinate system\n"
    "with the c0 axis aligned with the surface normal.  This surface\n"
    "is bounded by a circle.\n"
    "\n"
    "To create a SimCircPlanarSurf object:\n"
    "    __init__(pos,orient,radius)\n"
    "    ----\n\n"
    "    pos         # Position of object origin in 3-space.\n"
    "    orient      # Euler angles [phi,theta,psi] [rad].\n"
    "    radius      # Bounding circle radius.\n\n"
    "    ----\n",  /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    SimCircPlanarSurf_methods,  /* tp_methods */
    0,                         /* tp_members */
    SimCircPlanarSurf_getseters, /* tp_getset */
    &SimSurfType,              /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)SimCircPlanarSurf_init, /* tp_init */
    0,                         /* tp_alloc */
    PyType_GenericNew,         /* tp_new */
};

/////////////////////////////////////////////////////////////////
//
// class SimRectPlanarSurf
//
typedef struct {
  PyObject_HEAD
  SimSurf bSimSurf;
  PyArrayObject *bitmap;
  double prm[2];
  double pxorigin[2]; // Origin for pixel 0,0 in the c1-c2-plane.
  double dpc[2];      // Pixel size along c1, c2.
  int    npc[2];      // No. pixels along c1, c2.
} SimRectPlanarSurf;

static void SimRectPlanarSurf_dealloc(SimRectPlanarSurf *self) {
  cleanup_po(1,self->bitmap);
  self->ob_type->tp_free((PyObject *)self);
}

static int
SimRectPlanarSurf_init(SimRectPlanarSurf *self,
                        PyObject *args,PyObject *kwds) {
  PyObject *pos, *orient, *dargs;

  self->bitmap = NULL;

  if (!PyArg_ParseTuple(args,"OO(dd)",
                        &pos,&orient,&self->prm[0],&self->prm[1]))
    return -1;

  dargs = Py_BuildValue("OO",pos,orient);
  if ( SimSurfType.tp_init((PyObject *)self,dargs,kwds) ) {
    cleanup_po(1,dargs);
    return -1;
  }

  cleanup_po(1,dargs);

  return 0;
}

static PyObject *
SimRectPlanarSurf_get_prm(SimRectPlanarSurf *self) {
  return Py_BuildValue("dd",self->prm[0],self->prm[1]);
}

static PyObject *
SimRectPlanarSurf_setBitmap(SimRectPlanarSurf *self,PyObject *args) {
  PyObject      *obitmap;
  PyArrayObject *bitmap;
  int i;

  if (!PyArg_ParseTuple(args,"O",&obitmap))
    return NULL;

  bitmap = (PyArrayObject *)
    PyArray_ContiguousFromAny(obitmap,PyArray_DOUBLE,2,2);
  if ( bitmap == NULL )
    return NULL;

  for ( i = 0; i < 2; i++ ) {
    self->npc[i] = PyArray_DIMS(bitmap)[i];
    self->dpc[i] = 2.*self->prm[i]/self->npc[i];
  }

  self->pxorigin[0] =  self->prm[0] -self->dpc[0]/2.;
  self->pxorigin[1] = -self->prm[1] +self->dpc[1]/2.;

  self->bitmap = bitmap;

  Py_RETURN_NONE;
}

static PyObject *
SimRectPlanarSurf_intensity(SimRectPlanarSurf *self,PyObject *args) {
  PyObject        *mod, *pxshift, *res, *rval, *dargs;
  PyArrayObject   *ares;
  SimIntersection *insc;
  SimRay          *sray, *liray;

  double fpix[2], pxsa[2], spnt[3], *ptr_ares;
  int    ipix[2], i;

  if ( self->bitmap == NULL )
    return SimSurf_intensity((SimSurf *)self,args);

  if (!PyArg_ParseTuple(args,"OO",&insc,&liray))
    return NULL;

  sray = insc->sray;
  SimRay_point_core(sray,insc->t,spnt);

  for ( i = 0; i < 2; i++ ) {
    fpix[i] = (spnt[i+1] -self->pxorigin[i])/(self->dpc[i]);
    fpix[i] = fabs(fpix[i]);
    ipix[i] = (int)floor(fpix[i]);
    pxsa[i] = ipix[i] -fpix[i];

    if ( (fpix[i] < 1.) || (fpix[i] >= self->npc[i] -1) )
      return Py_BuildValue("d",0.);
  }
  PyRun_SimpleString("import sys;sys.path.append('/home/xbao/.conda/envs/piv1.0/lib/python2.7/site-packages/spivet/pivlib')");
  mod     = PyImport_ImportModule("pivutil");
  pxshift = PyObject_GetAttrString(mod,"pxshift");

  dargs = Py_BuildValue("O[[ii]][[dd]]",
                        self->bitmap,ipix[0],ipix[1],pxsa[0],pxsa[1]);
  res   = PyObject_CallObject(pxshift,dargs);
  Py_DECREF(dargs);

  ares     = (PyArrayObject *)PyArray_ContiguousFromAny(res,PyArray_DOUBLE,1,1);
  ptr_ares = (double *)PyArray_DATA(ares);
  rval     = Py_BuildValue("d",ptr_ares[0]);

  cleanup_po(4,mod,pxshift,res,ares);
  return rval;
}

static PyObject *
SimRectPlanarSurf_intersect(SimRectPlanarSurf *self,PyObject *args) {
  SimRay *lray;
  double *lsource, *lhead, *prm;

  double ic[2], t;
  int    ptc;

  if (!PyArg_ParseTuple(args,"O",&lray))
    return NULL;

  ptc     = 3*(lray->pointcount -1);
  lsource = lray->points +ptc;
  lhead   = lray->heads +ptc;

  prm     = self->prm;

  // Check for intersections.
  if ( fabs(lhead[0]) < PIVSIMC_DEPS ) {
    // Ray is parallel to the surface.
    Py_RETURN_NONE;
  }

  t = -lsource[0]/lhead[0];
  if ( t < 0. ) {
    // Surface is behind source and ray propagating away.
    Py_RETURN_NONE;
  }

  ic[0] = lhead[1]*t +lsource[1];
  ic[1] = lhead[2]*t +lsource[2];

  if ( ( fabs(ic[0]) > prm[0] ) || ( fabs(ic[1]) > prm[1] ) )
    Py_RETURN_NONE;

  return Py_BuildValue("[d]",t);
}

static PyObject *
SimRectPlanarSurf_normal(SimRectPlanarSurf *self,PyObject *args) {
  PyArrayObject *nrml;
  double *ptr_nrml;
  npy_intp dim = 3;

  nrml = (PyArrayObject *)PyArray_EMPTY(1,&dim,PyArray_DOUBLE,PyArray_CORDER);
  if (nrml == NULL )
    return NULL;

  ptr_nrml = (double *)PyArray_DATA(nrml);

  ptr_nrml[0] = 1.;
  ptr_nrml[1] = 0.;
  ptr_nrml[2] = 0.;

  return PyArray_Return(nrml);
}

static PyGetSetDef SimRectPlanarSurf_getseters[] = {
  {"prm",(getter)SimRectPlanarSurf_get_prm,NULL,
   "Returns halflengths [c1,c2].",NULL},
  {NULL},
};

static PyMethodDef SimRectPlanarSurf_methods[] = {
  {"intensity",(PyCFunction)SimRectPlanarSurf_intensity,METH_VARARGS,
   "intensity(insc,liray)\n"
   "----\n\n"
   "insc         # SimIntersection object for viewing ray.\n"
   "liray        # Illumination ray in local coordinates (c0,c1,c2).\n\n"
   "----\n\n"
   "Returns the interpolated pixel value for the stored image, if any.\n"
   "Returns Lambertian intensity otherwise (see SimSurf.intensity()).\n"
  },
  {"intersect",(PyCFunction)SimRectPlanarSurf_intersect,METH_VARARGS,
   "intersect(lray)\n"
   "----\n\n"
   "lray        # Ray in local coordinates (c0,c1,c2).\n\n"
   "----\n\n"
   "Returns a list containing the distance along lray.head from lray.source\n"
   "to the single surface intersection, if any. Returns None otherwise.\n"
  },
  {"normal",(PyCFunction)SimRectPlanarSurf_normal,METH_VARARGS,
   "normal(lpnt)\n"
   "----\n\n"
   "lpnt    # Point in local coordinate system (c0,c1,c2).\n\n"
   "----\n\n"
   "Computes the unit surface normal at the local surface point\n"
   "pnt (c0,c1,c2).\n"
   "\n"
   "For this surface, the returned normal vector will always\n"
   "point along the surface local c0-axis regardless of lpnt.\n"
  },
  {"setBitmap",(PyCFunction)SimRectPlanarSurf_setBitmap,METH_VARARGS,
   "setBitmap(bitmap)\n"
   "----\n\n"
   "bitmap      # mxn array containing the image values.\n\n"
   "----\n\n"
   "The SimRectPlanarSurf can store a bitmap and use that bitmap to\n"
   "determine the intensity of any ray striking the surface instead\n"
   "of treating the surface as Lambertian.  The surface is divided\n"
   "into the same number of pixels as the bitmap.\n"
  },
  {NULL},
};

static PyTypeObject SimRectPlanarSurfType = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "pivsimc.SimRectPlanarSurf", /*tp_name*/
    sizeof(SimRectPlanarSurf), /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)SimRectPlanarSurf_dealloc, /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    0,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    "The planar surface is centered around a local coordinate system\n"
    "with the c0 axis aligned with the surface normal.  This surface\n"
    "is bounded by a rectangle.\n"
    "\n"
    "A SimRectPlanarSurf can store a bitmap grayscale image.  The\n"
    "image will be scaled to fit the entire surface.  Pixel 0,0 will\n"
    "correspond to (prm[0],-prm[1]).  The image's first axis will be\n"
    "mapped to c1, and the second axis will be mapped to c2.  If a bitmap\n"
    "is stored, then a call to intensity() will return the interpolated\n"
    "bitmap value, otherwise the Lambertian intensity will be returned.\n"
    "\n"
    "The rectangle will have a size of 2*c1 x 2*c2.\n"
    "\n"
    "To create a SimRectPlanarSurf object:\n"
    "    __init__(pos,orient,prm)\n"
    "    ----\n\n"
    "    pos         # Position of object origin in 3-space.\n"
    "    orient      # Euler angles [phi,theta,psi] [rad].\n"
    "    prm         # Half lengths [c1, c2].\n\n"
    "    ----\n",  /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    SimRectPlanarSurf_methods,  /* tp_methods */
    0,                         /* tp_members */
    SimRectPlanarSurf_getseters, /* tp_getset */
    &SimSurfType,              /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)SimRectPlanarSurf_init, /* tp_init */
    0,                         /* tp_alloc */
    PyType_GenericNew,         /* tp_new */
};


/////////////////////////////////////////////////////////////////
//
// class SimSphericalSurf
//
typedef struct {
  PyObject_HEAD
  SimSurf   bSimSurf;
  double    radius;
  double    rsq;
} SimSphericalSurf;

static void SimSphericalSurf_dealloc(SimSphericalSurf *self) {
  self->ob_type->tp_free((PyObject *)self);
}

static int
SimSphericalSurf_init(SimSphericalSurf *self,
                        PyObject *args,PyObject *kwds) {
  PyObject *pos, *orient, *dargs;

  if (!PyArg_ParseTuple(args,"OOd",
                        &pos,&orient,&self->radius))
    return -1;

  self->rsq = self->radius*self->radius;

  dargs = Py_BuildValue("OO",pos,orient);
  if ( SimSurfType.tp_init((PyObject *)self,dargs,kwds) ) {
    cleanup_po(1,dargs);
    return -1;
  }

  cleanup_po(1,dargs);

  return 0;
}

static PyObject *
SimSphericalSurf_get_radius(SimSphericalSurf *self) {
  return Py_BuildValue("d",self->radius);
}

// The following method follows that of Akenine-Moller-2002,
// Section 13.5.2.
static PyObject *
SimSphericalSurf_intersect(SimSphericalSurf *self,PyObject *args) {
  PyObject *otv, *dargs;
  SimRay   *lray;
  double   *lsource, *lhead, *lv, *hv;

  double ls, s, ms, q, ptv[2];
  int    ptc;

  if (!PyArg_ParseTuple(args,"O",&lray))
    return NULL;

  ptc     = 3*(lray->pointcount -1);
  lsource = lray->points +ptc;
  lhead   = lray->heads +ptc;

  // Check for an intersection
  lv = lsource;
  PIVSIMC_VDOTV(lv,lv,ls);

  hv = lhead;
  PIVSIMC_VDOTV(lv,hv,s);

  if ( ( s > 0. ) && ( ls > self->rsq ) ) {
    // Sphere is behind ray source and ray is propagating away.
    Py_RETURN_NONE;
  }

  ms = ls -s*s;

  if ( ms > self->rsq ) {
    //Ray overshot sphere.
    Py_RETURN_NONE;
  }

  // We have an intersection.
  q      = sqrt(self->rsq -ms);
  ptv[1] = -1.;
  if ( ls > self->rsq ) {
    ptv[0] = -(s +q);
    if ( q > PIVSIMC_DEPS )
      ptv[1] = -(s -q);
  }
  else
    ptv[0] = q -s;

  // Package the results.
  otv = PyList_New(1);
  PyList_SetItem(otv,0,PyFloat_FromDouble(ptv[0]));
  if ( ptv[1] > 0. ) {
    dargs = PyFloat_FromDouble(ptv[1]);
    PyList_Append(otv,dargs);
    Py_DECREF(dargs);
  }

  return otv;
}

static PyObject *
SimSphericalSurf_normal(SimSphericalSurf *self,PyObject *args) {
  PyArrayObject *nrml;
  double *ptr_nrml;
  double lpnt[3];
  npy_intp dim = 3;

  if (!PyArg_ParseTuple(args,"(ddd)",&lpnt[0],&lpnt[1],&lpnt[2]))
    return NULL;

  nrml = (PyArrayObject *)PyArray_EMPTY(1,&dim,PyArray_DOUBLE,PyArray_CORDER);
  if (nrml == NULL )
    return NULL;

  ptr_nrml = (double *)PyArray_DATA(nrml);

  ptr_nrml[0] = lpnt[0]/self->radius;
  ptr_nrml[1] = lpnt[1]/self->radius;
  ptr_nrml[2] = lpnt[2]/self->radius;

  return PyArray_Return(nrml);
}

static PyGetSetDef SimSphericalSurf_getseters[] = {
  {"radius",(getter)SimSphericalSurf_get_radius,NULL,
   "Returns radius.",NULL},
  {NULL},
};

static PyMethodDef SimSphericalSurf_methods[] = {
  {"intersect",(PyCFunction)SimSphericalSurf_intersect,METH_VARARGS,
   "intersect(lray)\n"
   "----\n\n"
   "lray        # Ray in local coordinates (c0,c1,c2).\n\n"
   "----\n\n"
   "A ray can intersect a cylidrical surface twice, so a list\n"
   "of all intersection distances along lray.head is returned.  The \n"
   "list is ordered with the closest intersection listed first.  If only\n"
   "one intersection is found, then a list containing one entry will be\n"
   "returned.  If no intersections are found, returns None.\n"
  },
  {"normal",(PyCFunction)SimSphericalSurf_normal,METH_VARARGS,
   "normal(lpnt)\n"
   "----\n\n"
   "lpnt    # Point in local coordinate system (c0,c1,c2).\n\n"
   "----\n\n"
   "Computes the unit surface normal at the local surface point\n"
   "pnt (c0,c1,c2).\n"
   "\n"
   "NOTE:  lpnt should be on the surface as the computed normal\n"
   "vector is normalized by the radius.\n"
  },
  {NULL},
};

static PyTypeObject SimSphericalSurfType = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "pivsimc.SimSphericalSurf", /*tp_name*/
    sizeof(SimSphericalSurf), /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)SimSphericalSurf_dealloc, /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    0,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    "The spherical surface is centered around a local coordinate system.\n"
    "\n"
    "To create a SimSphericalSurf object:\n"
    "    __init__(pos,orient,prm)\n"
    "    ----\n\n"
    "    pos         # Position of object origin in 3-space.\n"
    "    orient      # Euler angles [phi,theta,psi] [rad].\n"
    "    radius      # Sphere radius.\n\n"
    "    ----\n\n", /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    SimSphericalSurf_methods,  /* tp_methods */
    0,                         /* tp_members */
    SimSphericalSurf_getseters, /* tp_getset */
    &SimSurfType,              /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)SimSphericalSurf_init, /* tp_init */
    0,                         /* tp_alloc */
    PyType_GenericNew,         /* tp_new */
};

////////////////////////////////////////////////////////////////
//
// class SimRefractiveObject
//
typedef struct {
  PyObject_HEAD
  SimObject bSimObject;
  PyObject  *ior;
  PyObject  *surfs;
} SimRefractiveObject;

static void SimRefractiveObject_dealloc(SimRefractiveObject *self) {
  cleanup_po(2,self->ior,self->surfs);
  self->ob_type->tp_free((PyObject *)self);
}

static int
SimRefractiveObject_init(SimRefractiveObject *self,
                         PyObject *args,PyObject *kwds) {
  PyObject *pos, *orient, *dargs;

  if (!PyArg_ParseTuple(args,"OOOO",&pos,&orient,&self->ior,&self->surfs))
    return -1;
  Py_INCREF(self->ior);
  Py_INCREF(self->surfs);

  dargs = Py_BuildValue("OO",pos,orient);
  if ( SimObjectType.tp_init((PyObject *)self,dargs,kwds) ) {
    cleanup_po(1,dargs);
    return -1;
  }

  cleanup_po(1,dargs);

  return 0;
}

static PyObject *SimRefractiveObject_get_n(SimRefractiveObject *self) {
  Py_INCREF(self->ior);
  return self->ior;
}

static PyObject *SimRefractiveObject_get_surfs(SimRefractiveObject *self) {
  Py_INCREF(self->surfs);
  return self->surfs;
}

static PyObject *
SimRefractiveObject_intensity(SimRefractiveObject *self, PyObject *args) {
  PyObject        *insc,*liray,*surf,*siray,*rval;

  if (!PyArg_ParseTuple(args,"OO",&insc,&liray))
    return NULL;

  surf = PyList_GetItem(self->surfs,((SimIntersection *)insc)->sndx);
  Py_INCREF(insc);
  Py_INCREF(liray);
  Py_INCREF(surf);

  siray = PyObject_CallMethod(surf,"parent2local","Oi",liray,0);

  rval  = PyObject_CallMethod(surf,"intensity","OO",insc,siray);
  cleanup_po(4,insc,liray,surf,siray);

  if (PyErr_Occurred()) goto error;

  return rval;

 error:
  cleanup_po(1,rval);
  return NULL;
}

static PyObject *
SimRefractiveObject_intersect(SimRefractiveObject *self,PyObject *args) {
  SimRay          *lray=NULL, *sray=NULL;
  SimSurf         *surf=NULL;
  SimIntersection *insc=NULL;
  PyObject        *tl=NULL, *ilst=NULL, *tv=NULL;
  PyObject        *dargs=NULL, **ptr_ailst;
  PyArrayObject   *spnt=NULL, *snml=NULL, *lnml=NULL, *atl=NULL, *srt=NULL;
  PyArrayObject   *ailst=NULL;
  double          *ptr_snml;
  int             *ptr_srt;

  double   t, d;
  int      nsurfs, i, j, ntv, exflg;
  npy_intp ntl;

  if (!PyArg_ParseTuple(args,"O",&lray))
    return NULL;
  Py_INCREF(lray);


  tl   = PyList_New(0);
  ilst = PyList_New(0);

  nsurfs  = PyList_Size(self->surfs);
  for ( i = 0; i < nsurfs; i++ ) {
    surf = (SimSurf *)PyList_GetItem(self->surfs,i);
    Py_INCREF(surf);

    // sray = surf.parent2local(lray)
    sray  = (SimRay *)PyObject_CallMethod((PyObject *)surf,"parent2local",
                                          "Oi",lray,0);

    // tv = surf.intersect(sray)
    tv = PyObject_CallMethod((PyObject *)surf,"intersect","O",sray);

    if ( tv == Py_None ) {
      Py_DECREF(surf);
      Py_DECREF(tv);
      Py_DECREF(sray);
      continue;
    }

    ntv = PyList_Size(tv);
    for ( j = 0; j < ntv; j++ ) {
      t = PyFloat_AsDouble( PyList_GetItem(tv,j) );

      // spnt = sray.point(t)
      dargs = Py_BuildValue("(d)",t);
      spnt  = (PyArrayObject *)SimRay_point((SimRay *)sray,dargs);
      Py_DECREF(dargs);

      // snml = surf.normal(spnt)
      snml  = (PyArrayObject *)PyObject_CallMethod((PyObject *)surf,
                                                   "normal",
                                                   "O",spnt);
      Py_DECREF(spnt);

      // lnml = surf.local2parent(snml,1)
      lnml  = (PyArrayObject *)PyObject_CallMethod((PyObject *)surf,
                                                   "local2parent","Oi",snml,1);

      // sray should have a pointcount of 1.
      // d = dot(sray.head,snml)
      ptr_snml = (double *)PyArray_DATA(snml);
      PIVSIMC_VDOTV(sray->heads,ptr_snml,d);
      if ( d < 0. )
        exflg = 0;
      else
        exflg = 1;

      // PyList_Append() increments the reference count, so we need to
      // dispose of our objects.
      dargs = PyFloat_FromDouble(t);
      PyList_Append(tl, dargs );
      Py_DECREF(dargs);

      dargs = Py_BuildValue("OiOOdOOi",self,i,lray,sray,t,lnml,snml,exflg);
      insc  = (SimIntersection *)PyObject_New(SimIntersection,
                                              &SimIntersectionType);

      SimIntersection_init(insc,dargs,NULL);
      PyList_Append(ilst, (PyObject *)insc );
      cleanup_po(4,dargs,lnml,snml,insc);

    }
    cleanup_po(3,tv,sray,surf);
  }
  Py_DECREF(lray);

  if (PyErr_Occurred()) goto error;

  ntl = PyList_Size(tl);
  if ( ntl == 0 ) {
    cleanup_po(2,tl,ilst);
    Py_RETURN_NONE;
  }

  atl   = (PyArrayObject *)PyArray_ContiguousFromAny(tl,PyArray_OBJECT,1,1);
  ailst =
    (PyArrayObject *)PyArray_EMPTY(1,&ntl,PyArray_OBJECT,PyArray_CORDER);
  if ( ( atl == NULL ) || ( ailst == NULL ) ) goto error;
  ptr_ailst = (PyObject **)PyArray_DATA(ailst);

  srt     = (PyArrayObject *)PyArray_ArgSort(atl,0,PyArray_QUICKSORT);
  ptr_srt = (int *)PyArray_DATA(srt);
  for ( i = 0; i < ntl; i++ ) {
    ptr_ailst[i] = PyList_GetItem(ilst,ptr_srt[i]);
    Py_INCREF(ptr_ailst[i]);
  }

  cleanup_po(4,tl,atl,ilst,srt);
  return PyArray_Return(ailst);  // Must return an array.

 error:
  cleanup_po(4,tl,ilst,atl,ailst);
  return NULL;
}

static PyObject *
SimRefractiveObject_setn(SimRefractiveObject *self,PyObject *args) {
  if (!PyArg_ParseTuple(args,"O",&self->ior))
    return NULL;

  Py_INCREF(self->ior);
  Py_RETURN_NONE;
}

static PyGetSetDef SimRefractiveObject_getseters[] = {
  {"n",(getter)SimRefractiveObject_get_n,NULL,
   "Returns refractive index.",NULL},
  {"surfs",(getter)SimRefractiveObject_get_surfs,NULL,
   "Returns surface list.",NULL},
  {NULL}
};

static PyMethodDef SimRefractiveObject_methods[] = {
  {"intensity",(PyCFunction)SimRefractiveObject_intensity,METH_VARARGS,
   "intensity(insc,liray)\n"
   "----\n\n"
   "insc        # SimIntersection object for intersecting ray.\n"
   "liray       # Ray from SimLight in local coordinates of this object.\n\n"
   "----\n\n"
   "Returns intensity for intersected surface.  See SimSurf.intensity() for\n"
   "more details."
  },
  {"intersect",(PyCFunction)SimRefractiveObject_intersect,METH_VARARGS,
   "intersect(lray)\n"
   "----\n\n"
   "lray        # SimRay object expressed in local coordinate system.\n\n"
   "----\n\n"
   "Intersects a ray with each surface of an object and returns\n"
   "a list of SimIntersection objects, ordered with increasing\n"
   "distance.  If no intersections are found, returns None.\n"
   "\n"
   "NOTE: lray must be expressed in the local coordinate system of\n"
   "the refractive object.\n"
  },
  {"setn",(PyCFunction)SimRefractiveObject_setn,METH_VARARGS,
   "setn(n)\n"
   "----\n\n"
   "n           # Refractive index.\n\n"
   "----\n\n"
   "Updates refractive index."
  },
  {NULL}
};

static PyTypeObject SimRefractiveObjectType = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "pivsimc.SimRefractiveObject",          /*tp_name*/
    sizeof(SimRefractiveObject),            /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)SimRefractiveObject_dealloc, /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    0,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    "Base class for a simulation object with constant refractive index.\n"
    "\n"
    "To create a SimRefractiveObject object:\n"
    "   __init__(pos,orient,n,surfs)\n"
    "   ----\n\n"
    "   pos         # Position of object origin in 3-space.\n"
    "   orient      # Euler angles [phi,theta,psi] [rad].\n"
    "   n           # Refractive index.\n"
    "   surfs       # List of bounding surfaces.\n\n"
    "   ----\n\n"
    "A SimRefractiveObject must be initialized with a postion (pos),\n"
    "orientation (orient), a refractive index (n), and a list of\n"
    "bounding surfaces (surfs).\n", /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    SimRefractiveObject_methods, /* tp_methods */
    0,                         /* tp_members */
    SimRefractiveObject_getseters, /* tp_getset */
    &SimObjectType,            /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)SimRefractiveObject_init, /* tp_init */
    0,                         /* tp_alloc */
    PyType_GenericNew,         /* tp_new */
};

////////////////////////////////////////////////////////////////
//
// Python prototype:
//   SimEnv_refract(ni,nf,insc)
//
// Refracts a ray at the intersection with surf.  Internal
// reflection can occur.
//
// Does not check for ni or nf = None.
//
// Returns refracted ray in SimEnv coordinates.
//
static PyObject *
SimEnv_refract(PyObject *oni,PyObject *onf,SimIntersection *insc) {
  int i;
  double ni = PyFloat_AsDouble(oni);
  double nf = PyFloat_AsDouble(onf);

  // It's easy to send in a bad refractive index.
  if (PyErr_Occurred()) return NULL;

  // If the two refractive indices are equal, return lray in the parent
  // coordinates.
  if ( fabs(ni -nf) < PIVSIMC_DEPS ) {
    PyObject *rray = PyObject_CallMethod(insc->obj,"local2parent",
                                         "Oi",insc->lray,0);
    return rray;
  }

  // Transform into surface local coordinates.
  PyObject *iobj  = insc->obj;
  PyObject *surfs = ((SimRefractiveObject *)iobj)->surfs;
  SimSurf  *surf  = (SimSurf *)PyList_GetItem(surfs,insc->sndx);
  SimRay   *sray  = insc->sray;
  double   *snml  = insc->snml;

  double *ptr_shead = sray->heads +3*(sray->pointcount -1);

  double spnt[3];
  SimRay_point_core(sray,insc->t,spnt);

  double d, ad, dd;
  PIVSIMC_VDOTV(ptr_shead,snml,d);
  ad = fabs(d);
  dd = 1. -ad;
  // If ad < PIVSIMC_DEPS, ray is tangent to surface.
  // If dd < PIVSIMC_DEPS, ray is perpendicular to surface.
  if ( ( ad < PIVSIMC_DEPS ) || ( dd < PIVSIMC_DEPS ) ) {
    // Ray is tangent to surface.
    sray = (SimRay *)PyObject_New(SimRay,&SimRayType);
    if ( SimRay_init_core((SimRay *)sray,spnt,ptr_shead) ) {
      Py_XDECREF(sray);
      return NULL;
    }

    PyObject *lray = PyObject_CallMethod((PyObject *)surf,"local2parent",
                                         "Oi",sray,0);
    PyObject *pray = PyObject_CallMethod(iobj,"local2parent","Oi",lray,0);

    Py_DECREF(lray);
    Py_DECREF(sray);
    return pray;
  }

  // Transform into the refraction coordinate system (r0, r1, r2).
  // r0 will be normal to the plane of refraction, r2 will be aligned
  // with surface normal, and r1 will be oriented such that
  // dot(r1,head) > 0.
  double s2rmat[9], *r1;
  for ( i = 0; i < 3; i++ ) {
    s2rmat[i +6] = snml[i];                  // r2
    s2rmat[i +3] = ptr_shead[i] -d*snml[i];  // r1
  }

  r1 = s2rmat +3;
  PIVSIMC_VDOTV(r1,r1,d);   // d = dot(r1,r1)
  d = 1./sqrt(d);
  for ( i = 0; i < 3; i++ )
    s2rmat[i +3] = s2rmat[i +3]*d;

  PIVSIMC_REVCROSS(snml,r1,s2rmat);  // r0
  PIVSIMC_VDOTV(s2rmat,s2rmat,d);    // d = dot(r0,r0)
  d = 1./sqrt(d);
  for ( i= 0; i < 3; i++ )
    s2rmat[i] = s2rmat[i]*d;

  double rhead[3];
  PIVSIMC_MVMUL(s2rmat,ptr_shead,rhead);

  // Refract the ray.
  double nisia = ni*rhead[1];  // ni*sin(alpha_incident)  (Always positive)
  if ( nisia < 0. ) // DEBUG
    PySys_WriteStdout("DEBUG: NISIA NEG\n");

  if ( nisia >= nf ) {
    // Total internal reflection.
    rhead[2]  = -rhead[2];
    insc->tir = 1;
  }
  else {
    double sra = nisia/nf;
    rhead[2]   = copysign(sqrt(1. -sra*sra),rhead[2]);
    rhead[1]   = sra;
  }

  PIVSIMC_MTVMUL(s2rmat,rhead,ptr_shead);
  sray = (SimRay *)PyObject_New(SimRay,&SimRayType);
  if ( SimRay_init_core((SimRay *)sray,spnt,ptr_shead) ) {
    Py_XDECREF(sray);
    return NULL;
  }

  PyObject *lray = PyObject_CallMethod((PyObject *)surf,"local2parent",
                                       "Oi",sray,0);
  PyObject *pray = PyObject_CallMethod(iobj,"local2parent","Oi",lray,0);

  Py_DECREF(lray);
  Py_DECREF(sray);
  return pray;
}


////////////////////////////////////////////////////////////////
//
// Python prototype:
//   SimEnv_refract_wrap(ni,nf,insc)
//
// This is a wrapper around SimEnv_refract() that allows
// the function to be called from within python.
//
// Refracts a ray at the intersection with surf.  Internal
// reflection can occur.
//
// Does not check for ni or nf = None.
//
// Returns refracted ray in SimEnv coordinates.
//
static PyObject *
SimEnv_refract_wrap(PyObject *self,PyObject *args) {
  PyObject *ni, *nf, *insc;

  if (!PyArg_ParseTuple(args,"OOO",&ni,&nf,&insc))
    return NULL;

  return SimEnv_refract(ni,nf,(SimIntersection *)insc);
}


////////////////////////////////////////////////////////////////
//
// Python prototype:
//   SimEnv_image(env,rays,bitmap,maxrcnt)
//
// Images a scene using rays from a camera and stores the image
// in bitmap.
//
// bitmap must be a contiguous numpy array.
//
static PyObject *
SimEnv_image(PyObject *self,PyObject *args) {
  PyObject *env, *rays, *ray;
  PyArrayObject *bitmap;
  double *ptr_bitmap;

  int maxrcnt, px, i, tpc;

  if (!PyArg_ParseTuple(args,"OOOi",&env,&rays,&bitmap,&maxrcnt))
    return NULL;

  // Initialization.
  ptr_bitmap  = (double *)PyArray_DATA(bitmap);
  int mxpxndx = PyArray_SIZE(bitmap) -1;
  mxpxndx = ( mxpxndx == 0 ? 1 : mxpxndx );

  PyObject *rayiter = PyObject_GetIter(rays);
  if ( rayiter == NULL )
    return NULL;

  PyObject *light = PyObject_GetAttrString(env,"m_light");
  PyObject *iray  = PyObject_GetAttrString(light,"iray");
  Py_DECREF(light);

  // Trace each ray.
  px  = 0;
  tpc = 10;
  while ( ( ray = PyIter_Next(rayiter) ) ) {
    PyObject *inode = PyObject_GetAttrString(env,"m_octree");
    PyObject *cobj  = Py_None;
    PyObject *nstk  = PyList_New(0);

    PyList_Append(nstk,((SimRefractiveObject *)env)->ior);
    int rcnt  = 0;
    while ( 1 ) {
      // Find nearest leaf node intersection.
      double lntv     = DBL_MAX;
      PyObject *lnisc = Py_None;
      Py_INCREF(lnisc);

      PyObject *leaves = PyObject_GetAttrString(inode,"leaves");

      int lfcnt = PyList_Size(leaves);
      for ( i = 0; i < lfcnt; i++ ) {
        PyObject *leaf = PyList_GetItem(leaves,i);
        PyObject *obj  = PyObject_GetAttrString(leaf,"obj");
        PyObject *lray = PyObject_CallMethod(obj,"parent2local",
                                             "Oi",ray,0);
        PyObject *isc  = PyObject_CallMethod(obj,"intersect","O",lray);

        if ( isc == Py_None ) {
          Py_DECREF(isc);
          Py_DECREF(lray);
          Py_DECREF(obj);
          continue;
        }

        PyObject *mnisc = ((PyObject **)PyArray_DATA(isc))[0];
        double    mnt   = ((SimIntersection *)mnisc)->t;
        if ( ( mnt < PIVSIMC_DEPS ) && ( obj == cobj ) ) {
          // Intersected leaf at ray.source, so set 'nearest' intersection
          // to the second one (if available).
          if ( PyArray_SIZE(isc) == 1 ) {
            Py_DECREF(isc);
            Py_DECREF(lray);
            Py_DECREF(obj);
            continue;
          }
          mnisc = ((PyObject **)PyArray_DATA(isc))[1];
          mnt   = ((SimIntersection *)mnisc)->t;
        }

        if ( mnt < lntv ) {
          lntv  = mnt;
          Py_DECREF(lnisc);
          lnisc = mnisc;
          Py_INCREF(lnisc);
        }

        Py_DECREF(isc);
        Py_DECREF(lray);
        Py_DECREF(obj);
      }  // for ( i = 0; i < lfcnt; i++ )

      Py_DECREF(leaves);

      // Find nearest inner node intersection.  The parent node must
      // be checked before self to enable crawling back up the tree.
      // Parent also needs to be checked because large objects generate
      // leaf nodes for several internal nodes.  A ray may cross the
      // parent internal node bounding box before intersecting the
      // surface of a large object.
      PyObject *inil     = PyList_New(0);
      PyObject *children = PyObject_GetAttrString(inode,"children");
      PyObject *inisc;

      int chldcnt = PyList_Size(children);
      for ( i = 0; i < chldcnt; i++ ) {
        PyObject *child = PyList_GetItem(children,i);

        inisc = PyObject_CallMethod(child,"intersect","O",ray);
        if ( inisc == Py_None ) {
          Py_DECREF(inisc);
          continue;
        }

        PyList_Append(inil,inisc);
        Py_DECREF(inisc);
      }
      Py_DECREF(children);

      PyObject *parent = PyObject_GetAttrString(inode,"parent");
      if ( parent != Py_None ) {
        inisc = PyObject_CallMethod(parent,"intersect","O",ray);
        if ( inisc != Py_None )
          PyList_Append(inil,inisc);

        Py_DECREF(inisc);
      }

      inisc = PyObject_CallMethod(inode,"intersect","O",ray);
      if ( inisc != Py_None )
        PyList_Append(inil,inisc);
      Py_DECREF(inisc);

      // When leaving, the ray will only intersect the SimEnv object.
      if (  ( parent == Py_None )
            && ( PyList_Size(inil) <= 1 )
            && ( ((SimIntersection *)lnisc)->obj == env ) ) {
        // Advance ray and leave environment.
        PyObject *dargs = Py_BuildValue("(d)",lntv);
        PyObject *rval  = SimRay_addSegment((SimRay *)ray,dargs);

        Py_XDECREF(rval);
        Py_DECREF(dargs);
        Py_DECREF(parent);
        Py_DECREF(inil);
        Py_DECREF(lnisc);
        break;
      }
      Py_DECREF(parent);

      double intv = DBL_MAX;
      int   inndx = -1;
      int leninil = PyList_Size(inil);
      for ( i = 0; i < leninil; i++ ) {
        PyObject *isc = ((PyObject **)PyArray_DATA(PyList_GetItem(inil,i)))[0];
        if ( ((SimIntersection *)isc)->t < intv ) {
          intv  = ((SimIntersection *)isc)->t;
          inndx = i;
        }
      }

      inisc = ((PyObject **)PyArray_DATA(PyList_GetItem(inil,inndx)))[0];

      // If the closest intersection intersection is an interior node,
      // just advance the ray.
      if ( intv < lntv ) {
        PyObject *dargs = Py_BuildValue("(d)",intv);
        PyObject *rval  = SimRay_addSegment((SimRay *)ray,dargs);

        Py_XDECREF(rval);
        Py_DECREF(dargs);

        Py_DECREF(inode);
        inode = ((SimIntersection *)inisc)->obj;
        Py_INCREF(inode);

        Py_DECREF(inil);
        Py_DECREF(lnisc);

        continue;
      }

      // We have intersected a refractive object in the present internal
      // node.
      PyObject *dargs = Py_BuildValue("(d)",lntv);
      PyObject *rval  = SimRay_addSegment((SimRay *)ray,dargs);
      Py_XDECREF(rval);
      Py_DECREF(dargs);
      if ( ((SimIntersection *)lnisc)->obj == cobj ) {
        if ( rcnt < maxrcnt )
          rcnt = rcnt +1;
        else {
          Py_DECREF(inil);
          Py_DECREF(lnisc);
          break;
        }
      }
      else {
        cobj = ((SimIntersection *)lnisc)->obj;
        rcnt = 0;
      }

      // Get the refractive indices.  If the refractive index of the
      // intersected object is None, then get its intensity.  Otherwise,
      // determine which refractive index corresponds to the incident
      // and refracted materials.
      //
      // Refractive indices are stored in a stack, nstk.  As a ray
      // enters a leaf node, the refractive index of that node is
      // appended to nstk.  When a ray exits an object, the last
      // refractive index on the stack is removed.
      SimIntersection     *io = (SimIntersection *)lnisc;
      SimRefractiveObject *ro = (SimRefractiveObject *)io->obj;
      PyObject            *n  = ro->ior;
      if ( n == Py_None ) {
        PyObject *liray = PyObject_CallMethod((PyObject *)ro,
                                              "parent2local",
                                              "Oi",iray,0);
        PyObject *gsv = PyObject_CallMethod((PyObject *)ro,
                                            "intensity",
                                            "OO",lnisc,liray);
        ptr_bitmap[px] = PyFloat_AsDouble(gsv);

        Py_DECREF(gsv);
        Py_DECREF(liray);

        Py_DECREF(inil);
        Py_DECREF(lnisc);

        break;
      }

      PyObject *ni, *nf;  // ni: incident, nf: refracted
      int endx = PyList_GET_SIZE(nstk) -1;
      if ( io->exflg ) {
        ni = PyList_GetItem(nstk,endx);
        if ( ni != n )
          PySys_WriteStdout("Refractive index mismatch.\n");  // DEBUG

        ni = n;

        endx = PyList_GET_SIZE(nstk) -2;
        nf   = PyList_GetItem(nstk,endx);
      }
      else {
        ni = PyList_GetItem(nstk,endx);
        nf = n;

        PyList_Append(nstk,nf);
      }

      // Refract.
      PyObject *rray   = SimEnv_refract(ni,nf,io);
      if (PyErr_Occurred()) return NULL;  // This leaks.
      PyObject *rrhead = SimRay_get_head((SimRay *)rray);

      dargs = Py_BuildValue("(O)",rrhead);
      rval  = SimRay_changeHeading((SimRay *)ray,dargs);

      // If the exit flag was set and total internal reflection did not
      // occur, pop the last entry from nstk.
      if ( io->exflg ) {
        if ( ! io->tir ) {
          endx = PyList_GET_SIZE(nstk) -1;
          PyObject *tnstk = PyList_GetSlice(nstk,0,endx);
          Py_DECREF(nstk);
          nstk = tnstk;
        }
      }

      Py_XDECREF(rval);
      Py_DECREF(dargs);
      Py_DECREF(rrhead);
      Py_DECREF(rray);

      Py_DECREF(inil);
      Py_DECREF(lnisc);
    } // Inner while.

    int prcc = (100*px)/mxpxndx;
    if ( prcc > tpc ) {
      tpc = (prcc/10)*10;
      PySys_WriteStdout(" | %i%% complete\n",tpc);
      tpc = tpc +10;
    }

    px = px +1;

    Py_DECREF(nstk);
    Py_DECREF(inode);
    Py_DECREF(ray);
  } // Outer while.

  cleanup_po(2, rayiter,iray);
  if (PyErr_Occurred())
    return NULL;

  Py_RETURN_NONE;
}

/////////////////////////////////////////////////////////////////
//
// WORKER FUNCTION cleanup_po
// Decrements references to PyObjects.
//
static void cleanup_po(int nvars, ...) {
  int i;
  va_list ap;
  PyObject *po;

  va_start(ap,nvars);
  for (i=0; i<nvars; i++) {
    po = va_arg(ap, PyObject *);
    if ( po == NULL )
      continue;
    else
      Py_DECREF(po);
  }
  va_end(ap);
}

static PyMethodDef pivsimc_methods[] = {
  {"SimEnv_refract_wrap",SimEnv_refract_wrap,METH_VARARGS,"Refracts a ray."},
  {"SimEnv_image",SimEnv_image,METH_VARARGS,"Forms image for a camera."},
  {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initpivsimc(void) {
  PyObject *mod;

  if (PyType_Ready(&SimRayType) < 0)
    return;
  if (PyType_Ready(&SimIntersectionType) < 0)
    return;
  if (PyType_Ready(&SimObjectType) < 0 )
    return;
  if (PyType_Ready(&SimSurfType) < 0 )
    return;
  if (PyType_Ready(&SimCylindricalSurfType) < 0 )
    return;
  if (PyType_Ready(&SimCircPlanarSurfType) < 0 )
    return;
  if (PyType_Ready(&SimRectPlanarSurfType) < 0 )
    return;
  if (PyType_Ready(&SimSphericalSurfType) < 0 )
    return;
  if (PyType_Ready(&SimRefractiveObjectType) < 0 )
    return;

  mod = Py_InitModule("pivsimc",pivsimc_methods);

  Py_INCREF(&SimRayType);
  PyModule_AddObject(mod,"SimRay",(PyObject *)&SimRayType);

  Py_INCREF(&SimIntersectionType);
  PyModule_AddObject(mod,"SimIntersection",(PyObject *)&SimIntersectionType);

  Py_INCREF(&SimObjectType);
  PyModule_AddObject(mod,"SimObject",(PyObject *)&SimObjectType);

  Py_INCREF(&SimSurfType);
  PyModule_AddObject(mod,"SimSurf",(PyObject *)&SimSurfType);

  Py_INCREF(&SimCylindricalSurfType);
  PyModule_AddObject(mod,"SimCylindricalSurf",
                     (PyObject *)&SimCylindricalSurfType);

  Py_INCREF(&SimCircPlanarSurfType);
  PyModule_AddObject(mod,"SimCircPlanarSurf",
                     (PyObject *)&SimCircPlanarSurfType);

  Py_INCREF(&SimRectPlanarSurfType);
  PyModule_AddObject(mod,"SimRectPlanarSurf",
                     (PyObject *)&SimRectPlanarSurfType);

  Py_INCREF(&SimSphericalSurfType);
  PyModule_AddObject(mod,"SimSphericalSurf",(PyObject *)&SimSphericalSurfType);

  Py_INCREF(&SimRefractiveObjectType);
  PyModule_AddObject(mod,"SimRefractiveObject",
                     (PyObject *)&SimRefractiveObjectType);

  import_array();
}

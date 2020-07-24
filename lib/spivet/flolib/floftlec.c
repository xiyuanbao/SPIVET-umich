/*
Filename:  floftlec.c
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
  These functions handle computationally expensive aspects of
  FTLE computation.

*/

#include "Python.h"
#include "numpy/arrayobject.h"

#include <math.h>
#include <stdlib.h>
#include <float.h>
#include <mkl_lapack.h>
typedef MKL_INT lpk_int;
//#include <clapack.h>

//typedef __CLPK_integer lpk_int;

//
// WORKER FUNCTION PROTOTYPES
//
static void bld_dphidd(double*,double*,npy_intp*,npy_intp*,npy_intp,npy_intp*,
                       double*);
static void cleanup_po(int nvars, ...);
static int compare_doubles(const void *,const void *);
static long svd_dphidd(double*,double*,double*);
//~

//
// MACROS
//

// Convert an index for a 1D array into a group of indices for the 3D array.
#define NDX_TO_INDICES(ndx,ncells,kndx,jndx,indx) \
    if (1) { \
        kndx = (ndx)/(ncells[1]*ncells[2]); \
        jndx = (ndx -(kndx)*ncells[1]*ncells[2])/ncells[2]; \
        indx = ndx -(kndx)*ncells[1]*ncells[2] -(jndx)*ncells[2]; \
    } else (void)0

// Convert a group of indices for a 3D array into an index for the 1D array.
#define INDICES_TO_NDX(ndx,ncells,kndx,jndx,indx) \
    if (1) { \
        ndx = (kndx)*ncells[1]*ncells[2] +(jndx)*ncells[2] +indx; \
    } else (void)0

// Computes l-1 and l+1 indices for a given dimension while compensating
// for Type 1 end cases (see ftlecore()).
#define TYPE1_NDC(dim,cndc,ncells,dcndc) \
    if (1) { \
        cndc[2*dim]    = (dcndc == 0 ? dcndc : dcndc -1); \
        cndc[2*dim +1] = (dcndc == ncells[dim] -1 ? dcndc : dcndc +1); \
    } else (void)0

// Computes l-1 and l+1 indices for a given dimension while compensating
// for Type 2 end cases (see ftlecore()).
#define TYPE2_NDC(dim,cndc,ptr_c2tndx,cndx,dcndc) \
    if (1) { \
        cndc[2*dim]    = (ptr_c2tndx[cndx[2*dim]] < 0 ? dcndc : cndc[2*dim]); \
        cndc[2*dim +1] = (ptr_c2tndx[cndx[2*dim +1]] < 0 ? dcndc : cndc[2*dim +1]); \
    } else (void)0



/////////////////////////////////////////////////////////////////
//
// Python prototype:
//   nc2tndx = bldc2tndx(t2cndx,vtmsk,ncells)
//
// ----
//
// t2cndx      Tracer to cell index map for the current timestep.
// vtmsk       Valid tracer mask for the current timestep.
// ncells      Number of cells in the underlying mesh [z,y,x].
//
// ----
//
// Builds nc2tndx, the new c2tndx valid for the next timestep.  nc2tndx
// will be set to the tracer index for the cell's tracer in the tcrd
// array of the next timestep provided the cell's tracer has not advected
// beyond the perimeter.  If the cell's tracer has advected beyond the
// perimeter, nc2tndx will be set to -1 for that cell.
//
// t2cndx and vtmsk are 1D arrays of length equal to the number of tracers
// for the current timestep.  nc2tndx is a 3D array of the same shape as
// the underlying mesh.
//
static PyObject *floftlec_bldc2tndx(PyObject *self, PyObject *args) {
    PyObject *ot2cndx, *ovtmsk;

    PyArrayObject *t2cndx, *vtmsk, *nc2tndx;

    npy_intp *ptr_t2cndx, *ptr_vtmsk;
    npy_intp *ptr_nc2tndx;

    int ncells[3];

    npy_intp cntr, i, cndx, ntr0, vtcnt;
    npy_intp npy_ncells[3];

    if ( !PyArg_ParseTuple(
                           args,
                           "OO(iii):bldc2tndx",
                           &ot2cndx, &ovtmsk, ncells, ncells +1, ncells +2
                           )
       )
        return NULL;

    t2cndx = (PyArrayObject *)PyArray_ContiguousFromAny(ot2cndx,
                                                        NPY_INTP,
                                                        1,1);
    vtmsk = (PyArrayObject *)PyArray_ContiguousFromAny(ovtmsk,
                                                        NPY_INTP,
                                                        1,1);

    if ( ( t2cndx == NULL ) || ( vtmsk  == NULL ) ) {
        cleanup_po(2,t2cndx,vtmsk);
        return NULL;
    }

    // Initialization.
    cntr = PyArray_DIMS(vtmsk)[0];
    ntr0 = ncells[0]*ncells[1]*ncells[2];

    for ( i = 0; i < 3; i++ ) {
        npy_ncells[i] = ncells[i];
    }

    nc2tndx = (PyArrayObject *)PyArray_EMPTY(3,npy_ncells,NPY_INTP,
                                             PyArray_CORDER);

    if ( nc2tndx == NULL ) {
        cleanup_po(2,t2cndx,vtmsk);
        return NULL;
    }

    ptr_t2cndx  = (npy_intp *)PyArray_DATA(t2cndx);
    ptr_vtmsk   = (npy_intp *)PyArray_DATA(vtmsk);
    ptr_nc2tndx = (npy_intp *)PyArray_DATA(nc2tndx);

    // Initialize nc2tndx.
    for ( i = 0; i < ntr0; i++ ) {
        ptr_nc2tndx[i] = -1;
    }

    // Main loop.
    vtcnt = 0;
    for ( i = 0; i < cntr; i++ ) {
        if ( ptr_vtmsk[i] ) {
            cndx              = ptr_t2cndx[i];
            ptr_nc2tndx[cndx] = vtcnt++;
        }
    }

    cleanup_po(2,t2cndx,vtmsk);
    return PyArray_Return(nc2tndx);
}



/////////////////////////////////////////////////////////////////
//
// Python prototype:
//   ftlecore(tcrd0,tcrd,t2cndx,c2tndx,vtmsk,dt,ftle,sfac,mxsfo)
//
// ----
//
// tcrd0       Tracer array at TS0.
// tcrd        Tracer array at current timestep.
// t2cndx      Tracer to cell index map for the current timestep.
// c2tndx      Cell to tracer index map for the current timestep.
// vtmsk       Valid tracer mask for the current timestep.
// dt          Time between current timestep and TS0.
// ftle        FTLE array.
// sfac        Stretch factor array.
// mxsfo       Maximum stretch factor orientation array.
//
// ----
//
// Computes the FTLE for cells whose tracers will be moving beyond
// the flow domain between the current timestep and the next.
//
// c2tndx must be set < 0 for cells not having valid tracers in the
// current timestep.
//
// tcrd0 is an nx3 array, where n is the number of tracers at TS0 (ie,
// the total number of cells in the underlying mesh).  tcrd is an mx3
// array, where m is the number of valid tracers at the current timestep.
// t2cndx and vtmsk are 1D arrays of length m.  c2tndx, ftle, sfac, and
// mxsfo are 3D arrays with the same shape as the underlying mesh.
//
static PyObject *floftlec_ftlecore(PyObject *self, PyObject *args) {
    PyObject *otcrd0, *otcrd, *ot2cndx, *oc2tndx, *ovtmsk, *oftle, *osfac;
    PyObject *omxsfo;
    double   dt;

    PyArrayObject *tcrd0, *tcrd, *t2cndx, *c2tndx, *vtmsk, *ftle, *sfac;
    PyArrayObject *mxsfo;

    double      *ptr_ftle, *ptr_sfac, *ptr_mxsfo, *ptr_tcrd0, *ptr_tcrd;
    npy_intp    *ptr_t2cndx, *ptr_c2tndx, *ptr_vtmsk;
    npy_intp    *ncells;

    double   dphidd[9], U[9], dd[3], mxcgev, pfac;
    npy_intp cntr, i, c, vcoset;

    long info;

    if ( !PyArg_ParseTuple(
                           args,
                           "OOOOOdOOO:ftlecore",
                           &otcrd0,&otcrd,&ot2cndx,&oc2tndx,&ovtmsk,
                           &dt,&oftle,&osfac,&omxsfo
                           )
       )
        return NULL;

    tcrd0   = (PyArrayObject *)PyArray_ContiguousFromAny(otcrd0,
                                                         PyArray_DOUBLE,
                                                         2,2);
    tcrd    = (PyArrayObject *)PyArray_ContiguousFromAny(otcrd,
                                                         PyArray_DOUBLE,
                                                         2,2);
    t2cndx  = (PyArrayObject *)PyArray_ContiguousFromAny(ot2cndx,
                                                         NPY_INTP,
                                                         1,1);
    c2tndx  = (PyArrayObject *)PyArray_ContiguousFromAny(oc2tndx,
                                                         NPY_INTP,
                                                         3,3);
    vtmsk   = (PyArrayObject *)PyArray_ContiguousFromAny(ovtmsk,
                                                         NPY_INTP,
                                                         1,1);
    ftle    = (PyArrayObject *)PyArray_ContiguousFromAny(oftle,
                                                         PyArray_DOUBLE,
                                                         3,3);
    sfac    = (PyArrayObject *)PyArray_ContiguousFromAny(osfac,
                                                         PyArray_DOUBLE,
                                                         4,4);
    mxsfo   = (PyArrayObject *)PyArray_ContiguousFromAny(omxsfo,
                                                         PyArray_DOUBLE,
                                                         4,4);


    if ( ( tcrd0   == NULL ) ||
         ( tcrd    == NULL ) ||
         ( t2cndx  == NULL ) ||
         ( c2tndx  == NULL ) ||
         ( vtmsk   == NULL ) ||
         ( ftle    == NULL ) ||
         ( sfac    == NULL ) ||
         ( mxsfo   == NULL ) ) {
        cleanup_po(8,tcrd0,tcrd,t2cndx,c2tndx,vtmsk,ftle,sfac,mxsfo);
        return NULL;
    }

    // Initialization.
    cntr   = PyArray_DIMS(vtmsk)[0];
    ncells = PyArray_DIMS(c2tndx);
    vcoset = ncells[0]*ncells[1]*ncells[2];

    pfac = 0.5/fabs(dt);

    ptr_tcrd0   = (double   *)PyArray_DATA(tcrd0);
    ptr_tcrd    = (double   *)PyArray_DATA(tcrd);
    ptr_t2cndx  = (npy_intp *)PyArray_DATA(t2cndx);
    ptr_c2tndx  = (npy_intp *)PyArray_DATA(c2tndx);
    ptr_vtmsk   = (npy_intp *)PyArray_DATA(vtmsk);
    ptr_ftle    = (double   *)PyArray_DATA(ftle);
    ptr_sfac    = (double   *)PyArray_DATA(sfac);
    ptr_mxsfo   = (double   *)PyArray_DATA(mxsfo);

    // FTLE loop.
    for ( i = 0; i < cntr; i++ ) {
        if ( ptr_vtmsk[i] ) {
            // Skip valid tracers.
            continue;
        }

        // Compute dphidd, the gradient of the flow map, and the SVD of
        // dphidd.
        bld_dphidd(ptr_tcrd0,ptr_tcrd,ptr_t2cndx,ptr_c2tndx,i,ncells,dphidd);

        info = svd_dphidd(dphidd,dd,U);
        if ( info != 0 ) {
            cleanup_po(8,tcrd0,tcrd,t2cndx,c2tndx,vtmsk,ftle,sfac,mxsfo);
            return PyErr_Format(PyExc_ArithmeticError,
                                "SVD computation failed : %i",
                                (int)info);
        }

        // Compute the max eigenvalue of the right Cauchy-Green tensor
        // (dphidd'*dphidd) and set FTLE.  We are counting on dgesvd_() to
        // return singular values sorted in order of decreasing value.
        mxcgev = ( dd[0] <= 0. ? 1. : dd[0]*dd[0] );  // Handles Type 3 scenario.
        ptr_ftle[ptr_t2cndx[i]] = pfac*log(mxcgev);

        // Store the stretch factors and the direction of maximum stretch.
        for ( c = 0; c < 3; c++ ) {
            ptr_sfac[c*vcoset +ptr_t2cndx[i]]  = dd[c];
            ptr_mxsfo[c*vcoset +ptr_t2cndx[i]] = U[c];
        }

    }  // for ( i = 0; i < cntr; i++ )

    cleanup_po(8,tcrd0,tcrd,t2cndx,c2tndx,vtmsk,ftle,sfac,mxsfo);
    Py_RETURN_NONE;
}


/////////////////////////////////////////////////////////////////
//
// Python prototype:
//   tavscore(tcrd0,tcrd,t2cndx,c2tndx,vtmsk,idt,tavsfac,itime)
//
// ----
//
// tcrd0       Tracer array at TS0.
// tcrd        Tracer array at current timestep.
// t2cndx      Tracer to cell index map for the current timestep.
// c2tndx      Cell to tracer index map for the current timestep.
// vtmsk       Valid tracer mask for the current timestep.
// idt         Time between current and previous timesteps.
// tavsfac     Time-averaged stretch factor array.  beta will be stored
//             in component zero and gamma in component 1.
// itime       Integration time array.
//
// ----
//
// Computes the time-averaged stretch factors for all cells.
//
// c2tndx must be set < 0 for cells not having valid tracers in the
// current timestep.
//
// tcrd0 is an nx3 array, where n is the number of tracers at TS0 (ie,
// the total number of cells in the underlying mesh).  tcrd is an mx3
// array, where m is the number of valid tracers at the current timestep.
// t2cndx and vtmsk are 1D arrays of length m.  c2tndx, tavsfac, and itime are
// 3D arrays with the same shape as the underlying mesh.
//
static PyObject *floftlec_tavscore(PyObject *self, PyObject *args) {
    PyObject *otcrd0, *otcrd, *ot2cndx, *oc2tndx, *ovtmsk, *otavsfac;
    PyObject *oitime;
    double   idt;

    PyArrayObject *tcrd0, *tcrd, *t2cndx, *c2tndx, *vtmsk, *tavsfac;
    PyArrayObject *itime;

    double    *ptr_tavsfac, *ptr_itime, *ptr_tcrd0, *ptr_tcrd;
    npy_intp  *ptr_t2cndx, *ptr_c2tndx, *ptr_vtmsk;
    npy_intp  *ncells;

    double   dphidd[9], U[9], dd[3];
    npy_intp cntr, i, c, vcoset;
    long     info;

    double eps  = 2.*DBL_EPSILON;

    if ( !PyArg_ParseTuple(
                           args,
                           "OOOOOdOO:tavscore",
                           &otcrd0,&otcrd,&ot2cndx,&oc2tndx,&ovtmsk,
                           &idt,&otavsfac,&oitime
                           )
       )
        return NULL;

    tcrd0   = (PyArrayObject *)PyArray_ContiguousFromAny(otcrd0,
                                                         PyArray_DOUBLE,
                                                         2,2);
    tcrd    = (PyArrayObject *)PyArray_ContiguousFromAny(otcrd,
                                                         PyArray_DOUBLE,
                                                         2,2);
    t2cndx  = (PyArrayObject *)PyArray_ContiguousFromAny(ot2cndx,
                                                         NPY_INTP,
                                                         1,1);
    c2tndx  = (PyArrayObject *)PyArray_ContiguousFromAny(oc2tndx,
                                                         NPY_INTP,
                                                         3,3);
    vtmsk   = (PyArrayObject *)PyArray_ContiguousFromAny(ovtmsk,
                                                         NPY_INTP,
                                                         1,1);
    tavsfac = (PyArrayObject *)PyArray_ContiguousFromAny(otavsfac,
                                                         PyArray_DOUBLE,
                                                         4,4);
    itime   = (PyArrayObject *)PyArray_ContiguousFromAny(oitime,
                                                         PyArray_DOUBLE,
                                                         3,3);


    if ( ( tcrd0   == NULL ) ||
         ( tcrd    == NULL ) ||
         ( t2cndx  == NULL ) ||
         ( c2tndx  == NULL ) ||
         ( vtmsk   == NULL ) ||
         ( tavsfac == NULL ) ||
         ( itime   == NULL ) ) {
        cleanup_po(7,tcrd0,tcrd,t2cndx,c2tndx,vtmsk,tavsfac,itime);
        return NULL;
    }

    // Initialization.
    cntr   = PyArray_DIMS(vtmsk)[0];
    ncells = PyArray_DIMS(c2tndx);
    vcoset = ncells[0]*ncells[1]*ncells[2];

    ptr_tcrd0   = (double   *)PyArray_DATA(tcrd0);
    ptr_tcrd    = (double   *)PyArray_DATA(tcrd);
    ptr_t2cndx  = (npy_intp *)PyArray_DATA(t2cndx);
    ptr_c2tndx  = (npy_intp *)PyArray_DATA(c2tndx);
    ptr_vtmsk   = (npy_intp *)PyArray_DATA(vtmsk);
    ptr_tavsfac = (double   *)PyArray_DATA(tavsfac);
    ptr_itime   = (double   *)PyArray_DATA(itime);

    // Main loop.
    for ( i = 0; i < cntr; i++ ) {
        // Compute dphidd, the gradient of the flow map, and the SVD of
        // dphidd.  We are counting on dgesvd_() to return the singular
        // values sorted in order of decreasing value.
        bld_dphidd(ptr_tcrd0,ptr_tcrd,ptr_t2cndx,ptr_c2tndx,i,ncells,dphidd);

        info = svd_dphidd(dphidd,dd,U);
        if ( info != 0 ) {
            cleanup_po(7,tcrd0,tcrd,t2cndx,c2tndx,vtmsk,tavsfac,itime);
            return PyErr_Format(PyExc_ArithmeticError,
                                "SVD computation failed : %i",
                                (int)info);
        }

        // Increment variables for average stretch ratio computation.  If
        // dd[2] = 0., assume we have a 2D scenario and manually set dd[2]
        // to preserve incompressible flow assumption.
        //
        // Note: qsort returns the stretch factors in order of increasing
        // value.
        dd[2] = ( dd[2] <= eps ? 1./(dd[0]*dd[1]) : dd[2] );

        qsort((void *)dd,3,sizeof(double),compare_doubles);

        dd[2] = dd[2]/dd[0];  // beta
        dd[1] = dd[1]/dd[0];  // gamma

        for ( c = 0; c < 2; c++ ) {
            ptr_tavsfac[c*vcoset +ptr_t2cndx[i]] =
                ptr_tavsfac[c*vcoset +ptr_t2cndx[i]] +idt*dd[2-c];
        }

        ptr_itime[ptr_t2cndx[i]] = ptr_itime[ptr_t2cndx[i]] +idt;

    }  // for ( i = 0; i < cntr; i++ )

    cleanup_po(7,tcrd0,tcrd,t2cndx,c2tndx,vtmsk,tavsfac,itime);
    Py_RETURN_NONE;
}


/////////////////////////////////////////////////////////////////
//
// WORKER FUNCTION bld_dphidd
// Constructs the gradient of the flow map.
//
static void bld_dphidd(double   *ptr_tcrd0,
                       double   *ptr_tcrd,
                       npy_intp *ptr_t2cndx,
                       npy_intp *ptr_c2tndx,
                       npy_intp  ctndx,
                       npy_intp *ncells,
                       double   *dphidd) {

    double   dd[3];
    npy_intp c, r, ckndx, cjndx, cindx, cndc[6], cndx[6];

    double eps  = 2.*DBL_EPSILON;

    NDX_TO_INDICES(ptr_t2cndx[ctndx],ncells,ckndx,cjndx,cindx);

    // Compute gradient of flow map using central differences.
    // Three end cases need to be handled:
    //   1) If the cell is on the outer perimeter of all cells (ie,
    //      edge of the flow domain), use a first order forward or
    //      backward difference.
    //   2) If the cell is not on the edge of the domain, but
    //      is instead on the perimeter of available tracers for
    //      the current step, also use a first order difference.
    //   3) The cell having a tracer is isolated on all sides by
    //      cells not having tracers.  This is a pathological case
    //      indicative of poor mesh or temporal resolution.  Set
    //      FTLE = 0 for this case.
    //
    // 2D conceptual overview taken at a particular timestep.
    // Lower case letters represent cells with tracers that have
    // moved beyond the domain.  d-cells line the domain perimeter.
    // i-cells are interior cells.  Three cases are depicted.
    // Case a-a and case b-b are of Type 2 above.  Case c-c is
    // of Type 1.
    //
    //     abc
    //   --------------
    //  c|dddDDDDDDDDD
    //  b|diIIIIIIIIII
    //  a|dIIIIIIIIIII
    //

    // z-axis.  Note that ptr_c2tndx will be < 0 for neighbors of
    // of a Type 2 cell that are beyond the perimeter of cells having
    // valid tracers.
    TYPE1_NDC(0,cndc,ncells,ckndx);
    INDICES_TO_NDX(cndx[0],ncells,cndc[0],cjndx,cindx);
    INDICES_TO_NDX(cndx[1],ncells,cndc[1],cjndx,cindx);
    TYPE2_NDC(0,cndc,ptr_c2tndx,cndx,ckndx);

    INDICES_TO_NDX(cndx[0],ncells,cndc[0],cjndx,cindx);  // k-1,j,i
    INDICES_TO_NDX(cndx[1],ncells,cndc[1],cjndx,cindx);  // k+1,j,i
    dd[0] = (ptr_tcrd0[3*cndx[1]] -ptr_tcrd0[3*cndx[0]]);
    dd[0] = ( fabs(dd[0]) <= eps ? 1. : 1./dd[0] );      // 1/dz

    // y-axis.
    TYPE1_NDC(1,cndc,ncells,cjndx);
    INDICES_TO_NDX(cndx[2],ncells,ckndx,cndc[2],cindx);
    INDICES_TO_NDX(cndx[3],ncells,ckndx,cndc[3],cindx);
    TYPE2_NDC(1,cndc,ptr_c2tndx,cndx,cjndx);

    INDICES_TO_NDX(cndx[2],ncells,ckndx,cndc[2],cindx);  // k,j-1,i
    INDICES_TO_NDX(cndx[3],ncells,ckndx,cndc[3],cindx);  // k,j+1,i
    dd[1] = (ptr_tcrd0[3*cndx[3]+1] -ptr_tcrd0[3*cndx[2]+1]);
    dd[1] = ( fabs(dd[1]) <= eps ? 1. : 1./dd[1] );      // 1/dy

    // x-axis.
    TYPE1_NDC(2,cndc,ncells,cindx);
    INDICES_TO_NDX(cndx[4],ncells,ckndx,cjndx,cndc[4]);
    INDICES_TO_NDX(cndx[5],ncells,ckndx,cjndx,cndc[5]);
    TYPE2_NDC(2,cndc,ptr_c2tndx,cndx,cindx);

    INDICES_TO_NDX(cndx[4],ncells,ckndx,cjndx,cndc[4]);  // k,j,i-1
    INDICES_TO_NDX(cndx[5],ncells,ckndx,cjndx,cndc[5]);  // k,j,i+1
    dd[2] = (ptr_tcrd0[3*cndx[5]+2] -ptr_tcrd0[3*cndx[4]+2]);
    dd[2] = ( fabs(dd[2]) <= eps ? 1. : 1./dd[2] );      // 1/dx

    /*
     * dphidd is:
     *  / dphiz/dz  dphiz/dy  dphiz/dx \
     *  | dphiy/dz  dphiy/dy  dphiy/dx |
     *  \ dphix/dz  dphix/dy  dphix/dx /
     *
     * Build it in column major mode.
     */
    for ( c = 0; c < 3; c++ ) {
        for ( r = 0; r < 3; r++ ) {
            dphidd[3*c +r] =
                dd[c]*( ptr_tcrd[ 3*ptr_c2tndx[cndx[c*2 +1]] +r ]
                       -ptr_tcrd[ 3*ptr_c2tndx[cndx[c*2   ]] +r ] );
        }
    }
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


/////////////////////////////////////////////////////////////////
//
// WORKER FUNCTION
//
// Compares two double precision floating point values.
// Returns:
//   -1: a < b
//    0: a == b
//    1: a > b
//
static int compare_doubles(const void *a, const void *b) {
  double da, db;

  da = *( (const double *)a );
  db = *( (const double *)b );

  return (da > db) - (da < db);
}


/////////////////////////////////////////////////////////////////
//
// WORKER FUNCTION svd_dphidd
// Compute the SVD of dphidd
//
static long svd_dphidd(double *dphidd, double *dd, double *U) {
    double work[30];

    double dmy  = 0.;

    lpk_int mdim = 3;
    lpk_int wlen = 30;  // Must match work.
    lpk_int one  = 1;
    lpk_int info = 0;

    dgesvd_("A","N",&mdim,&mdim,dphidd,&mdim,dd,U,&mdim,&dmy,&one,
           work,&wlen,&info);

    return info;
}


static PyMethodDef floftlec_methods[] = {
  {"bldc2tndx", floftlec_bldc2tndx, METH_VARARGS, "Builds c2tndx."},
  {"ftlecore", floftlec_ftlecore, METH_VARARGS, "Computes FTLE"},
  {"tavscore", floftlec_tavscore, METH_VARARGS,
   "Computes time-averaged stretch factors."},
  {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initfloftlec(void) {
  (void) Py_InitModule("floftlec",floftlec_methods);
  import_array();
}

/*
Filename:  tlclibc.c
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
  These functions are for the numerically intensive parts of
  TlcLIB.

*/


#include "Python.h"
#include "numpy/arrayobject.h"

#include <math.h>
#include <stdlib.h>
#include <float.h>

//
// MACROS
//

// double val,coeff[],rval
// int    porder,i
#define TLCLIBC_EVALUPOLY(val,coeff,porder,rval,i) \
    if (1) { \
        rval = coeff[porder]; \
        for (i=porder-1; i>=0; i--) \
            rval = rval*val +coeff[i]; \
    } else (void) 0

//
// WORKER FUNCTION PROTOTYPES
//
static void cleanup_po(int nvars, ...);
//~


/////////////////////////////////////////////////////////////////
//
// Python prototype:
//   tlclibc.evalmpoly(idvar,pcoeff,porder)
//
// Computes the polynomial for each point in idvar using the
// specified coefficients and polynomial order.
//
// Given that these functions are primarily intended for TlcLIB
// functionality, consider a polynomial of the form
//     T = T(z,theta,hue)
//       = sum( c[i,j,k] * z**(i) * theta**(j) * hue**(k) )
//
// idvar should be ordered as [m,n], where m is the number of points,
// and n is the number of variables (eg, 3 for z, theta, hue).
//
// pcoeff should be a 1D array of polynomial coefficients arranged
// such that the independent variable with the highest index (hue in
// the above example) varies quickest.
//
// porder should be a 1D array of the polynomial order (ie, degree)
// for each independent variable.
//
// Returns rva.
//
static PyObject *tlclibc_evalmpoly(PyObject *self, PyObject *args) {
    PyObject      *oidvar, *opcoeff, *oporder;
    PyArrayObject *idvar, *pcoeff, *porder, *rva;

    double   *ptr_idvar, *ptr_pcoeff, *ptr_rva, *mcoeff, *coeff;
    npy_intp *ptr_porder;

    double   vval, rval;
    npy_intp npoly, nvar, vndx, mcsz, pco, i, j, v, p;
    npy_intp npts;

    if (!PyArg_ParseTuple(
              args,
              "OOO:evalmpoly",
              &oidvar,&opcoeff,&oporder
              )
        )
      return NULL;

    idvar  = (PyArrayObject *)PyArray_ContiguousFromAny(oidvar,PyArray_DOUBLE,2,2);
    pcoeff = (PyArrayObject *)PyArray_ContiguousFromAny(opcoeff,PyArray_DOUBLE,1,1);
    porder = (PyArrayObject *)PyArray_ContiguousFromAny(oporder,NPY_INTP,1,1);

    if ( ( idvar == NULL ) || ( pcoeff == NULL ) || ( porder == NULL ) ) {
        cleanup_po(3,idvar,pcoeff,porder);
        return NULL;
    }

    // Initialization.
    npts = PyArray_DIMS(idvar)[0];
    nvar = PyArray_DIMS(idvar)[1];

    rva = (PyArrayObject *)PyArray_EMPTY(1,&npts,PyArray_DOUBLE,PyArray_CORDER);
    if ( rva == NULL ) {
        cleanup_po(3,idvar,pcoeff,porder);
        return NULL;
    }

    ptr_idvar  = (double   *)PyArray_DATA(idvar);
    ptr_pcoeff = (double   *)PyArray_DATA(pcoeff);
    ptr_porder = (npy_intp *)PyArray_DATA(porder);
    ptr_rva    = (double   *)PyArray_DATA(rva);

    mcsz = ptr_porder[0] +1;
    for (i=1; i<nvar-1; i++)
        mcsz = mcsz*(ptr_porder[i] +1);

    mcoeff = (double *)PyMem_Malloc(mcsz*sizeof(double));
    if ( mcoeff == NULL ) {
        cleanup_po(4,idvar,pcoeff,porder,rva);
        return NULL;
    }

    // Compute the polynomial.
    for (p=0; p<npts; p++) {
        coeff = ptr_pcoeff;
        for (v=0; v<nvar; v++) {
            vndx = nvar -v -1;
            vval = ptr_idvar[p*nvar +vndx];

            if ( vndx > 0 )
                npoly = ptr_porder[0] +1;
            else
                npoly = 1;

            for (j=1; j<vndx; j++)
                npoly = npoly*(ptr_porder[j] +1);

            for (i=0; i<npoly; i++) {
                pco = i*(ptr_porder[vndx] +1);
                TLCLIBC_EVALUPOLY(vval,(&coeff[pco]),ptr_porder[vndx],rval,j);
                mcoeff[i] = rval;
            }
            coeff = mcoeff;
        }
        ptr_rva[p] = coeff[0];
    }

    PyMem_Free(mcoeff);
    cleanup_po(3,idvar,pcoeff,porder);

    return PyArray_Return(rva);
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


static PyMethodDef tlclibc_methods[] = {
        {"evalmpoly", tlclibc_evalmpoly, METH_VARARGS,
         "Evaluates multivariate polynomial at points."
        },
        {NULL,NULL,0,NULL}
};

PyMODINIT_FUNC inittlclibc(void) {
    (void) Py_InitModule("tlclibc",tlclibc_methods);
    import_array();
}

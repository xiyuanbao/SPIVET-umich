/*
Filename:  flohetc.c
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
  the flohet module.

*/

#include "Python.h"
#include "numpy/arrayobject.h"

#include <math.h>
#include <stdlib.h>
#include <float.h>

//
// WORKER FUNCTION PROTOTYPES
//
static void cleanup_mm(int nvars, ...);
static void cleanup_po(int nvars, ...);
static void *get_scipy_fun(char*,char*);
static double seconv(double,double,double*,int*);
//~

/////////////////////////////////////////////////////////////////
//
// Python prototype:
//   map = bldmapcore(astbnds,nast,sbnds,ns,saxnc)
//
// ----
//
// astbnds     a* bounds.
// nast        Number of a* data points.
// sbnds       Semi-axis bounds.
// ns          Number of semi-axis data points.
// saxnc       3-element array of cell count for each semi-axis.
//
// ----
//
// Builds the a* = a*(conc,s1,s0) map.
//
// Returns [map,dc,corigin,ds,sorigin]
//
//
static PyObject *flohetc_bldmapcore(PyObject *self, PyObject *args) {
    double astbnds[2], sbnds[2], s[3], ds, dc, da, cmx;
    double cmn, cval, fp, dmy, xast;
    int    nast, ns, saxnc[3], i, j, k, mshape[3], oset[2], tpc, extrap;

    PyObject      *rlst;
    PyArrayObject *map;

    double *ptr_map, *time, *icvala, *fcvala, *asta, *wghts, *nta, *coefa;
    double *wrk;
    long   *iwrk;

    long   lnast, nest, nnts, lwrk, info, clen;
    long   lzero = 0;
    long   sdeg  = 3;
    double dzero = 0.;

    int (*curfit)(long *,long *,double *,double *,double *,double *,
                  double *,long *,double *,long *,long *,double *,
                  double *,double *,double *,long *,long *,long *);

    int (*splev)(double *,long *,double *,long *,double *,
                 double *,long *,long *);

    if ( !PyArg_ParseTuple(
                           args,
                           "(dd)i(dd)i(iii):bldmapcore",
                           &astbnds[0],&astbnds[1],&nast,
                           &sbnds[0],&sbnds[1],&ns,
                           &saxnc[0],&saxnc[1],&saxnc[2]
                           )
       )
        return NULL;

    // Get references to scipy's curfit and splev functions.
    curfit = get_scipy_fun("scipy.interpolate.dfitpack","curfit");
    splev  = get_scipy_fun("scipy.interpolate.dfitpack","splev");

    if ( ( curfit == NULL ) || ( splev == NULL ) ) {
        return PyErr_Format(PyExc_TypeError,
                            "Failed to find scipy curfit and/or splev.");
    }

    // Initialization.
    xast = -1.E6;  // Extrapolation a*.

    ds = (sbnds[1] -sbnds[0])/(ns -1.);
    da = (astbnds[1] -astbnds[0])/(nast -1.);

    time = (double *)PyMem_Malloc(nast*sizeof(double));
    for ( i=0; i<nast; i++ ) {
        dmy = i*da +astbnds[0];
        time[i] = 1./(dmy*dmy);
    }

    mshape[0] = nast;
    mshape[1] = mshape[2] = ns;

    map = (PyArrayObject *)PyArray_EMPTY(3,mshape,PyArray_DOUBLE,PyArray_CORDER);
    if ( map == NULL ) {
        Py_RETURN_NONE;
    }

    ptr_map = (double *)PyArray_DATA(map);

    icvala = (double *)PyMem_Malloc(nast*sizeof(double));
    fcvala = (double *)PyMem_Malloc(nast*sizeof(double));
    asta   = (double *)PyMem_Malloc(nast*sizeof(double));
    wghts  = (double *)PyMem_Malloc(nast*sizeof(double));

    // Compute first cut of the map.  Align a* with k-axis, s1 with j-axis,
    // and s0 with i-axis.
    cmn = 1.;
    cmx = 0.;
    tpc = 0;
    for ( k=0; k<nast; k++ ) {
        oset[0] = k*ns*ns;

        for ( j=0; j<ns; j++ ) {
            s[1] = j*ds +sbnds[0];

            oset[1] = j*ns;

            for ( i=0; i<ns; i++ ) {
                s[0] = i*ds +sbnds[0];
                s[2] = 1./(s[0]*s[1]);

                cval = seconv(time[k],1.,s,saxnc);
                ptr_map[oset[0] +oset[1] +i] = cval;

                cmn = ( cval < cmn ? cval : cmn );
                cmx = ( cval > cmx ? cval : cmx );
            }
        }
        if ( fmod(k,nast/10) == 0 ) {
            tpc = tpc +1;
            PySys_WriteStdout(" |-| %i%% complete\n",10*tpc);
        }
    }
    cmn = (cmn > 0. ? cmn : 0.);
    cmx = (cmx < 1. ? cmx : 1.);

    // At this point, the map holds concentration values for equal-spaced a*.
    // We need to convert the map so that it stores a* values with the k-axis
    // spanning uniformly discretized concentration.  To do this, we need to
    // spline the existing a* = a*(conc) relationship, and then interpolate
    // onto the uniformly spaced concentration grid.
    //
    // The spline operations depend on SciPy's fitpack.
    dc = (cmx -cmn)/(nast -1.);
    for ( i=0; i<nast; i++ ) {
        fcvala[i] = i*dc +cmn;        // Uniform concentration values.
        asta[i]   = i*da +astbnds[0]; // Uniform a* values.
        wghts[i]  = 1.;
    }

    lnast = nast;
    nest  = sdeg +lnast +1;
    lwrk  = lnast*(sdeg+1) +nest*(7 +3*sdeg);

    coefa = (double *)PyMem_Malloc(nest*sizeof(double));
    nta   = (double *)PyMem_Malloc(nest*sizeof(double));
    wrk   = (double *)PyMem_Malloc(lwrk*sizeof(double));
    iwrk  = (long *)PyMem_Malloc(nest*sizeof(long));

    for ( j=0; j<ns; j++ ) {
        oset[1] = j*ns;
        for ( i=0; i<ns; i++ ) {
            extrap = 0;

            // Extract non-uniformly spaced composition values from the
            // map.
            clen = 0;
            icvala[0] = ptr_map[oset[1] +i];
            for ( k=1; k<nast; k++ ) {
                oset[0]   = k*ns*ns;
                icvala[k] = ptr_map[oset[0] +oset[1] +i];

                if ( icvala[k] > icvala[k-1] ) {
                    clen = k;
                }
                else {
                    break;
                }
            }
            clen = clen +1;

            // Spline the data.
            (*curfit)(&lzero,&clen,icvala,asta,wghts,&icvala[0],&icvala[clen-1],
                      &sdeg,&dzero,&nest,&nnts,nta,coefa,&fp,wrk,&lwrk,iwrk,
                      &info);

            if ( info > 0 ) {
                // Spline failed.  Set sequence to xast.
                extrap = 1;
            }
            else {
                // Interpolate a* values onto uniform concentration grid.
                (*splev)(nta,&nnts,coefa,&sdeg,fcvala,icvala,&lnast,&info);
                if ( info > 0 ) {
                    cleanup_mm(9,time,icvala,fcvala,asta,wghts,coefa,nta,wrk,iwrk);
                    return PyErr_Format(PyExc_ArithmeticError,
                                          "Spline evaluation failed : %i",
                                          (int)info);
                }
            }

            // Copy a* values into the map.  During the first cut of the map,
            // time values were chosen for uniformly spaced a*.  The upper
            // limit of a* in astbnds is likely not high enough to achieve a
            // cmx concentration for very stretched ellipses.  Consequently
            // for these very stretched ellipses, obtaining cmx concentration
            // will require extrapolation, which is not desirable. If
            // a* > astbnds[1], set all remaining cells to xast to indicate
            // that extrapolation is occurring.  We want xast to be large
            // and negative so that if we interpolate into the map at a
            // later date, there will be no doubt when operating on the
            // border of valid values (the interpolation will return a
            // negative value for a*).
            for ( k=0; k<nast; k++ ) {
                oset[0] = k*ns*ns;

                if ( extrap ) {
                    ptr_map[oset[0] +oset[1] +i] = xast;
                }
                else {
                    dmy = icvala[k];
                    if ( dmy > astbnds[1] ) {
                        extrap = 1;
                        dmy    = xast;
                    }
                    dmy = ( dmy < 0. ? xast : dmy );

                    ptr_map[oset[0] +oset[1] +i] = dmy;
                }
            }
        }
    }

    cleanup_mm(9,time,icvala,fcvala,asta,wghts,coefa,nta,wrk,iwrk);

    rlst = PyList_New(5);
    PyList_SetItem(rlst,0,PyArray_Return(map));
    PyList_SetItem(rlst,1,Py_BuildValue("d",dc));
    PyList_SetItem(rlst,2,Py_BuildValue("d",cmn));
    PyList_SetItem(rlst,3,Py_BuildValue("d",ds));
    PyList_SetItem(rlst,4,Py_BuildValue("d",sbnds[0]));

    return rlst;
}


/////////////////////////////////////////////////////////////////
//
// Python prototype:
//   cconc = secconc(t,D,sax,saxnc)
//
// ----
//
// t           Time.
// D           Diffusivity.
// sax         3-element array providing the semi-axis lengths.
// saxnc       3-element array of cell count for each semi-axis.
//
// ----
//
// Python callable wrapper around seconv().  See seconv() for details.
//
// Returns cconc, the concentration at the center of the ellipsoid at time
// t.
//
static PyObject *flohetc_secconc(PyObject *self, PyObject *args) {
    double t, D, sax[3], cconc;
    int    saxnc[3];

    if ( !PyArg_ParseTuple(
                           args,
                           "dd(ddd)(iii):secconc",
                           &t,&D,&sax[0],&sax[1],&sax[2],
                           &saxnc[0],&saxnc[1],&saxnc[2]
                           )
       )
        return NULL;

    cconc = seconv(t,D,sax,saxnc);

    return Py_BuildValue("d",cconc);
}


/////////////////////////////////////////////////////////////////
//
// WORKER FUNCTION cleanup_mm
// Frees PyMem_Malloc'ed memory.
//
static void cleanup_mm(int nvars, ...) {
  int i;
  va_list ap;
  void *mm;

  va_start(ap,nvars);
  for (i=0; i<nvars; i++) {
    mm = va_arg(ap, void *);
    PyMem_Free(mm);
  }
  va_end(ap);
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
// WORKER FUNCTION get_scipy_fun
// Returns a void* pointer to the scipy function funstr in the
// the fully qualified module modstr.
//
static void *get_scipy_fun(char *modstr,char *funstr) {
    PyObject *mod, *fun, *cobj;
    void *ptr_fun;

    mod = PyImport_ImportModule(modstr);
    if ( mod == NULL ) {
        return NULL;
    }

    fun = PyObject_GetAttrString(mod,funstr);
    if ( fun == NULL ) {
        cleanup_po(1,mod);
        return NULL;
    }

    cobj = PyObject_GetAttrString(fun,"_cpointer");
    if ( cobj == NULL ) {
        cleanup_po(2,mod,fun);
        return NULL;
    }
    if ( ! PyCObject_Check(cobj) ) {
        cleanup_po(3,mod,fun,cobj);
        return NULL;
    }

    ptr_fun = PyCObject_AsVoidPtr(cobj);
    cleanup_po(3,mod,fun,cobj);

    return ptr_fun;
}


/////////////////////////////////////////////////////////////////
//
// WORKER FUNCTION seconv
//
// ----
//
// t           Time.
// D           Diffusivity.
// sax         3-element array providing the semi-axis lengths.
// saxnc       3-element array of cell count for each semi-axis.
//
// ----
//
// Computes the concentration at the center of an ellipsoid that has
// diffused for t time units.  The initial concentration field is assumed to
// be equal to 1.0 inside the ellipse and 0.0 outside the ellipse.  The
// concentration is then computed by convolving the 3D heat kernel with the
// initial conditions.  The 3D heat kernel is given by
//     1/pow(4*pi*D*t,1.5)*exp(-r^2/(4*D*t))
// with
//     r^2 = x^2 +y^2 +z^2
//
// The semi-axes lengths should be passed sorted in order of decreasing
// magnitude.
//
// Returns cconc, the concentration at the center of the ellipsoid at time
// t.
//
static double seconv(double t, double D, double *sax, int *saxnc) {
    double rsaxsq[3], rat[3], cconc, csz[3], cvol, crdsq[3], chk;
    double krnl, ksf, esf;
    int    axnc[3], cntrc[3], i, j, k;

    double pi = 3.14159265359;

    // Initialization.
    cvol = 1.;
    for ( i=0; i<3; i++ ) {
        axnc[i]   = 2*saxnc[i] +1;
        cntrc[i]  = axnc[i]/2;
        csz[i]    = 2.*sax[i]/axnc[i];
        rsaxsq[i] = 1./(sax[i]*sax[i]);

        cvol = cvol*csz[i];
    }

    ksf = cvol/pow(4.*pi*D*t,1.5);
    esf = -1./(4.*D*t);

    // Main loop.  Align k-axis with minor axis and i-axis with major.
    cconc = 0.;
    for ( k=0; k<axnc[2]; k++ ) {
        crdsq[2] = csz[2]*(k -cntrc[2]);
        crdsq[2] = crdsq[2]*crdsq[2];
        rat[2]   = rsaxsq[2]*crdsq[2];

        for ( j=0; j<axnc[1]; j++ ) {
            crdsq[1] = csz[1]*(j -cntrc[1]);
            crdsq[1] = crdsq[1]*crdsq[1];
            rat[1]   = rsaxsq[1]*crdsq[1];

            for ( i=0; i<axnc[1]; i++ ) {
                crdsq[0] = csz[0]*(i -cntrc[0]);
                crdsq[0] = crdsq[0]*crdsq[0];
                rat[0]   = rsaxsq[0]*crdsq[0];

                chk  = rat[0] +rat[1] +rat[2];
                krnl = ksf*exp( esf*(crdsq[0] +crdsq[1] +crdsq[2]) );
                krnl = ( chk <= 1. ? krnl : 0. );
                cconc = cconc +krnl;
            }
        }
    }

    return cconc;
}



static PyMethodDef flohetc_methods[] = {
  {"bldmapcore", flohetc_bldmapcore, METH_VARARGS,
   "Builds concentration map."},
  {"secconc", flohetc_secconc, METH_VARARGS,
   "Center composition for static ellipse."},
  {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initflohetc(void) {
  (void) Py_InitModule("flohetc",flohetc_methods);
  import_array();
}

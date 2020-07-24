/*
Filename:  ex2lib.c
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
  These functions are wrappers for the Exodus II data file
  format.

*/

#include "Python.h"
#include "numpy/arrayobject.h"

#include "exodusII.h"

#include <math.h>
#include <stdlib.h>
#include <stdarg.h>

enum err_codes {
  EX2ERR_Invalid_Argument = 0,
  EX2ERR_Allocation_Error,
  EX2ERR_Invalid_Array_Dimensions,
  EX2ERR_Inquire_Failed,
  EX2ERR_Library_Error
};

static char ex2err[5][30] = {
  "Invalid argument : %s",
  "Allocation error : %s",
  "Invalid array dimensions : %s",
  "Inquire failed : %s",
  "ExodusII library error : %i" };

static void cleanup_po(int nvars, ...);
static PyObject *fparray(PyObject *ob, int typen, int mnd, int mxd);
static PyObject *intarray(PyObject *ob, int mnd, int mxd);

//***************************************************************
// ex_create
//
static PyObject *ex2lib_ex_create(PyObject *self, PyObject *args) {
  char *path;
  int  cmode, comp_ws, io_ws, rval;

  if (!PyArg_ParseTuple(args,
                        "siii:ex_create",
                        &path, &cmode, &comp_ws, &io_ws)
      )
    return PyErr_Format(PyExc_TypeError,ex2err[EX2ERR_Invalid_Argument],"");

  rval = ex_create(path,cmode,&comp_ws,&io_ws);
  if ( rval < 0 )
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Library_Error],rval);

  return Py_BuildValue("iii",rval,comp_ws,io_ws);
}

//***************************************************************
// ex_open
//
static PyObject *ex2lib_ex_open(PyObject *self, PyObject *args) {
  char *path;
  int mode, comp_ws, io_ws=0, rval;
  float version;

  if (!PyArg_ParseTuple(args,
                        "sii:ex_open",
                        &path, &mode, &comp_ws)
      )
    return PyErr_Format(PyExc_TypeError,ex2err[EX2ERR_Invalid_Argument],"");


  rval = ex_open(path,mode,&comp_ws,&io_ws,&version);
  if ( rval < 0 )
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Library_Error],rval);

  return Py_BuildValue("iii",rval,comp_ws,io_ws);
}

//***************************************************************
// ex_close
//
static PyObject *ex2lib_ex_close(PyObject *self, PyObject *args) {
  int exoid, comp_ws, io_ws, rval;

  if (!PyArg_ParseTuple(args,
                        "(iii):ex_close",
                        &exoid,&comp_ws,&io_ws)
      )
    return PyErr_Format(PyExc_TypeError,ex2err[EX2ERR_Invalid_Argument],"");

  rval = ex_close(exoid);
  if ( rval < 0 )
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Library_Error],rval);

  return Py_BuildValue("i",rval);
}


//***************************************************************
// ex_update
//
static PyObject *ex2lib_ex_update(PyObject *self, PyObject *args) {
  int exoid, comp_ws, io_ws, rval;

  if (!PyArg_ParseTuple(args,
                        "(iii):ex_update",
                        &exoid,&comp_ws,&io_ws)
      )
    return PyErr_Format(PyExc_TypeError,ex2err[EX2ERR_Invalid_Argument],"");

  rval = ex_update(exoid);
  if ( rval < 0 )
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Library_Error],rval);

  return Py_BuildValue("i",rval);
}


//***************************************************************
// ex_get_init
//
static PyObject *ex2lib_ex_get_init(PyObject *self, PyObject *args) {
  int exoid, num_dim, num_nodes, num_elem, num_elem_blk, num_node_sets;
  int num_side_sets, comp_ws, io_ws, rval;
  char title[MAX_LINE_LENGTH+1];

  if (!PyArg_ParseTuple(args,
                        "(iii):ex_get_init",
                        &exoid,&comp_ws,&io_ws)
      )
    return PyErr_Format(PyExc_TypeError,ex2err[EX2ERR_Invalid_Argument],"");

  rval = ex_get_init(exoid, title, &num_dim, &num_nodes, &num_elem,
                     &num_elem_blk, &num_node_sets, &num_side_sets);

  if ( rval != 0 )
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Library_Error],rval);

  return Py_BuildValue("siiiiii",title,num_dim,num_nodes,num_elem,
                       num_elem_blk,num_node_sets,num_side_sets);
}


//***************************************************************
// ex_put_init
//
static PyObject *ex2lib_ex_put_init(PyObject *self, PyObject *args) {
  int exoid, num_dim, num_nodes, num_elem, num_elem_blk, num_node_sets;
  int num_side_sets, comp_ws, io_ws, rval;
  char *title;

  if (!PyArg_ParseTuple(args,
                        "(iii)siiiiii:ex_put_init",
                        &exoid, &comp_ws, &io_ws, &title, &num_dim,
                        &num_nodes, &num_elem, &num_elem_blk, &num_node_sets,
                        &num_side_sets)
      )
    return PyErr_Format(PyExc_TypeError,ex2err[EX2ERR_Invalid_Argument],"");

  rval = ex_put_init(exoid, title, num_dim, num_nodes, num_elem, num_elem_blk,
                     num_node_sets, num_side_sets);
  if ( rval < 0 )
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Library_Error],rval);

  return Py_BuildValue("i",rval);
}


//***************************************************************
// ex_get_qa
//
static PyObject *ex2lib_ex_get_qa(PyObject *self, PyObject *args) {
  int i, j, exoid, num_qa_records, comp_ws, io_ws, rval;
  float fdmy;
  char cdmy;
  npy_intp dim[2];
  PyObject *oqalst, *oqarec;
  char **qa_record;
  char *qa_strings;

  if (!PyArg_ParseTuple(args,
                        "(iii):ex_get_qa",
                        &exoid,&comp_ws,&io_ws)
      )
    return PyErr_Format(PyExc_TypeError,ex2err[EX2ERR_Invalid_Argument],"");

  rval = ex_inquire(exoid,EX_INQ_QA,&num_qa_records,&fdmy,&cdmy);
  if ( rval < 0 )
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Inquire_Failed],"");

  dim[0] = num_qa_records;
  dim[1] = 4;

  qa_strings = (char *)malloc(4*dim[0]*(MAX_STR_LENGTH+1)*sizeof(char));
  qa_record  = (char **)malloc(4*dim[0]*sizeof(char *));
  if ( ( qa_strings == NULL ) || ( qa_record == NULL ) ) {
    return PyErr_Format(PyExc_MemoryError,ex2err[EX2ERR_Allocation_Error],"");
  }

  for ( i = 0; i < dim[0]; i++ ) {
    for ( j = 0; j < 4 ; j++ ) {
      qa_record[4*i +j] = qa_strings +(4*i +j)*(MAX_STR_LENGTH+1);
    }
  }

  rval = ex_get_qa(exoid,(char *(*)[4])(qa_record));

  if ( rval < 0 ) {
    free(qa_record);
    free(qa_strings);
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Library_Error],rval);
  }

  oqalst = PyList_New(dim[0]);
  for ( i = 0; i < dim[0]; i++ ) {
    oqarec = PyList_New(4);
    for ( j = 0; j < 4; j++ )
      PyList_SetItem(oqarec,j,Py_BuildValue("s",qa_record[4*i +j]));

    PyList_SetItem(oqalst,i,oqarec);
  }

  free(qa_record);
  free(qa_strings);
  return oqalst;
}


//***************************************************************
// ex_put_qa
//
static PyObject *ex2lib_ex_put_qa(PyObject *self, PyObject *args) {
  int i, j, exoid, num_qa_records, comp_ws, io_ws, rval;
  npy_intp *dim;
  PyObject *oqa_record;
  PyArrayObject *aoqa_record;
  PyObject **ptr_sobj;
  char **qa_record;

  if (!PyArg_ParseTuple(args,
                        "(iii)iO:ex_put_qa",
                        &exoid, &comp_ws, &io_ws, &num_qa_records, &oqa_record)
      )
    return PyErr_Format(PyExc_TypeError,ex2err[EX2ERR_Invalid_Argument],"");

  aoqa_record = (PyArrayObject *)PyArray_ContiguousFromAny(oqa_record,
                                                           PyArray_OBJECT,2,2);
  if ( aoqa_record == NULL )
    return PyErr_Format(PyExc_MemoryError,ex2err[EX2ERR_Allocation_Error],"");

  dim = aoqa_record->dimensions;
  if ( ( dim[0] > 0 ) && ( dim[1] != 4 ) )
    return PyErr_Format(PyExc_ValueError,
			ex2err[EX2ERR_Invalid_Array_Dimensions],
			"");

  qa_record = (char **)malloc(4*dim[0]*sizeof(char *));
  if ( qa_record == NULL ) {
    cleanup_po(1,aoqa_record);
    return PyErr_Format(PyExc_MemoryError,ex2err[EX2ERR_Allocation_Error],"");
  }

  ptr_sobj = (PyObject **)PyArray_DATA(aoqa_record);

  for (i=0; i<dim[0]; i++) {
    for (j=0; j<4; j++) {
      qa_record[i*4 +j] = PyString_AsString( ptr_sobj[i*4 +j]  );
    }
  }

  rval = ex_put_qa(exoid,num_qa_records,(char *(*)[4])(qa_record));

  free(qa_record);
  cleanup_po(1,aoqa_record);
  if ( rval < 0 )
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Library_Error],rval);

  return Py_BuildValue("i",rval);
}


//***************************************************************
// ex_get_info
//
static PyObject *ex2lib_ex_get_info(PyObject *self, PyObject *args) {
  int exoid, num_info, comp_ws, io_ws, rval, i;
  float fdmy;
  char cdmy;
  PyObject *oinfo;
  char *infos, **info;

  if (!PyArg_ParseTuple(args,
                        "(iii):ex_get_info",
                        &exoid, &comp_ws, &io_ws)
      )
    return PyErr_Format(PyExc_TypeError,ex2err[EX2ERR_Invalid_Argument],"");

  rval = ex_inquire(exoid,EX_INQ_INFO,&num_info,&fdmy,&cdmy);
  if ( rval < 0 )
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Inquire_Failed],"");

  infos = (char *)malloc(num_info*(MAX_LINE_LENGTH+1)*sizeof(char));
  info  = (char **)malloc(num_info*sizeof(char *));
  if ( ( infos == NULL ) || ( info == NULL ) ) {
    return PyErr_Format(PyExc_MemoryError,ex2err[EX2ERR_Allocation_Error],"");
  }

  for ( i = 0; i < num_info; i++ ) {
      info[i] = infos +i*(MAX_LINE_LENGTH+1);
  }

  rval = ex_get_info(exoid,info);

  if ( rval < 0 ) {
    free(info);
    free(infos);
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Library_Error],rval);
  }

  oinfo = PyList_New(num_info);
  for ( i = 0; i < num_info; i++ )
    PyList_SetItem(oinfo,i,Py_BuildValue("s",info[i]));

  free(info);
  free(infos);
  return oinfo;
}


//***************************************************************
// ex_put_info
//
static PyObject *ex2lib_ex_put_info(PyObject *self, PyObject *args) {
  int exoid, num_info, rval, llen, i, comp_ws, io_ws;
  PyObject *oinfo;
  PyArrayObject *aoinfo;
  PyObject **ptr_sobj;
  char **info;

  if (!PyArg_ParseTuple(args,
                        "(iii)iO:ex_put_info",
                        &exoid, &comp_ws, &io_ws, &num_info, &oinfo)
      )
    return PyErr_Format(PyExc_TypeError,ex2err[EX2ERR_Invalid_Argument],"");

  aoinfo = (PyArrayObject *)PyArray_ContiguousFromAny(oinfo,
                                                      PyArray_OBJECT,1,1);
  if ( aoinfo == NULL )
    return PyErr_Format(PyExc_MemoryError,ex2err[EX2ERR_Allocation_Error],"");

  llen = (aoinfo->dimensions)[0];
  info = (char **)malloc(llen*sizeof(char *));
  if ( info == NULL ) {
    cleanup_po(1,aoinfo);
    return PyErr_Format(PyExc_MemoryError,ex2err[EX2ERR_Allocation_Error],"");
  }

  ptr_sobj = (PyObject **)PyArray_DATA(aoinfo);

  for (i=0; i<llen; i++) {
    info[i] = PyString_AsString( ptr_sobj[i] );
  }

  rval = ex_put_info(exoid,num_info,info);

  free(info);
  cleanup_po(1,aoinfo);
  if ( rval < 0 )
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Library_Error],rval);

  return Py_BuildValue("i",rval);
}


//***************************************************************
// ex_inquire
//
static PyObject *ex2lib_ex_inquire(PyObject *self, PyObject *args) {
  int exoid, comp_ws, io_ws, rval, ival, req_info;
  float fval;
  char cval[MAX_LINE_LENGTH+1];

  if (!PyArg_ParseTuple(args,
                        "(iii)i:ex_inquire",
                        &exoid,&comp_ws,&io_ws,&req_info)
      )
    return PyErr_Format(PyExc_TypeError,ex2err[EX2ERR_Invalid_Argument],"");

  rval = ex_inquire(exoid,req_info,&ival,&fval,cval);
  if ( rval != 0 )
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Library_Error],rval);

  return Py_BuildValue("ids",ival,fval,cval);
}


//***************************************************************
// ex_get_coord
//
static PyObject *ex2lib_ex_get_coord(PyObject *self, PyObject *args) {
  int exoid, rval, comp_ws, io_ws, num_nodes, fdtype;
  npy_intp npy_num_nodes;
  float fdmy;
  char cdmy;
  PyObject *rlst;
  PyArrayObject *xc, *yc, *zc;
  void *ptr_xc, *ptr_yc, *ptr_zc;

  if (!PyArg_ParseTuple(args,
                        "(iii):ex_get_coord",
                        &exoid, &comp_ws, &io_ws)
      )
    return PyErr_Format(PyExc_TypeError,ex2err[EX2ERR_Invalid_Argument],"");

  rval = ex_inquire(exoid,EX_INQ_NODES,&num_nodes,&fdmy,&cdmy);
  if ( rval < 0 )
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Inquire_Failed],"");

  if ( comp_ws == 4 )
    fdtype = PyArray_FLOAT32;
  else
    fdtype = PyArray_FLOAT64;

  npy_num_nodes = num_nodes;

  xc = (PyArrayObject *)PyArray_EMPTY(1,&npy_num_nodes,fdtype,PyArray_CORDER);
  yc = (PyArrayObject *)PyArray_EMPTY(1,&npy_num_nodes,fdtype,PyArray_CORDER);
  zc = (PyArrayObject *)PyArray_EMPTY(1,&npy_num_nodes,fdtype,PyArray_CORDER);

  if ( (xc == NULL) || (yc == NULL) || (zc == NULL) ) {
    cleanup_po(3,xc,yc,zc);
    return PyErr_Format(PyExc_MemoryError,ex2err[EX2ERR_Allocation_Error],"");
  }

  ptr_xc = (void *)PyArray_DATA(xc);
  ptr_yc = (void *)PyArray_DATA(yc);
  ptr_zc = (void *)PyArray_DATA(zc);

  rval = ex_get_coord(exoid,ptr_xc,ptr_yc,ptr_zc);
  if ( rval < 0 ) {
    cleanup_po(3,xc,yc,zc);
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Library_Error],rval);
  }

  rlst = PyList_New(3);
  PyList_SetItem(rlst,0,PyArray_Return(xc));
  PyList_SetItem(rlst,1,PyArray_Return(yc));
  PyList_SetItem(rlst,2,PyArray_Return(zc));
  return rlst;
}


//***************************************************************
// ex_put_coord
//
static PyObject *ex2lib_ex_put_coord(PyObject *self, PyObject *args) {
  int exoid, rval, comp_ws, io_ws, fdtype;
  PyObject *oxc, *oyc, *ozc;
  PyArrayObject *xc, *yc, *zc;
  void *ptr_xc, *ptr_yc, *ptr_zc;

  if (!PyArg_ParseTuple(args,
                        "(iii)OOO:ex_put_coord",
                        &exoid, &comp_ws, &io_ws, &oxc, &oyc, &ozc)
      )
    return PyErr_Format(PyExc_TypeError,ex2err[EX2ERR_Invalid_Argument],"");

  if ( comp_ws == 4 )
    fdtype = PyArray_FLOAT32;
  else
    fdtype = PyArray_FLOAT64;

  xc = (PyArrayObject *)fparray(oxc,fdtype,1,1);
  yc = (PyArrayObject *)fparray(oyc,fdtype,1,1);
  zc = (PyArrayObject *)fparray(ozc,fdtype,1,1);

  if ( (xc == NULL) || (yc == NULL) || (zc == NULL) ) {
    cleanup_po(3,xc,yc,zc);
    return PyErr_Format(PyExc_MemoryError,ex2err[EX2ERR_Allocation_Error],"");
  }

  ptr_xc = (void *)PyArray_DATA(xc);
  ptr_yc = (void *)PyArray_DATA(yc);
  ptr_zc = (void *)PyArray_DATA(zc);

  rval = ex_put_coord(exoid,ptr_xc,ptr_yc,ptr_zc);

  cleanup_po(3,xc,yc,zc);
  if ( rval < 0 )
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Library_Error],rval);

  return Py_BuildValue("i",rval);
}


//***************************************************************
// ex_get_coord_names
//
static PyObject *ex2lib_ex_get_coord_names(PyObject *self, PyObject *args) {
  int exoid, rval, i, comp_ws, io_ws;
  PyObject *oname;
  char *names, **name;

  if (!PyArg_ParseTuple(args,
                        "(iii):ex_get_coord_names",
                        &exoid, &comp_ws, &io_ws)
      )
    return PyErr_Format(PyExc_TypeError,ex2err[EX2ERR_Invalid_Argument],"");

  names = (char *)malloc(3*(MAX_STR_LENGTH+1)*sizeof(char));
  name  = (char **)malloc(3*sizeof(char *));
  if ( ( names == NULL ) || ( name == NULL ) ) {
    return PyErr_Format(PyExc_MemoryError,ex2err[EX2ERR_Allocation_Error],"");
  }

  for ( i = 0; i < 3; i++ ) {
      name[i] = names +i*(MAX_STR_LENGTH+1);
  }

  rval = ex_get_coord_names(exoid,name);

  if ( rval < 0 ) {
    free(name);
    free(names);
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Library_Error],rval);
  }

  oname = PyList_New(3);
  for ( i = 0; i < 3; i++ )
    PyList_SetItem(oname,i,Py_BuildValue("s",name[i]));

  free(name);
  free(names);
  return oname;
}


//***************************************************************
// ex_put_coord_names
//
static PyObject *ex2lib_ex_put_coord_names(PyObject *self, PyObject *args) {
  int exoid, rval, llen, i, comp_ws, io_ws;
  PyObject *ocoord_names;
  PyArrayObject *aocoord_names;
  PyObject **ptr_sobj;
  char **coord_names;

  if (!PyArg_ParseTuple(args,
                        "(iii)O:ex_put_coord_names",
                        &exoid, &comp_ws, &io_ws, &ocoord_names)
      )
    return PyErr_Format(PyExc_TypeError,ex2err[EX2ERR_Invalid_Argument],"");

  aocoord_names =
    (PyArrayObject *)PyArray_ContiguousFromAny(ocoord_names,
                                               PyArray_OBJECT,1,1);
  if ( aocoord_names == NULL )
    return PyErr_Format(PyExc_MemoryError,ex2err[EX2ERR_Allocation_Error],"");

  llen = (aocoord_names->dimensions)[0];
  coord_names = (char **)malloc(llen*sizeof(char *));
  if ( coord_names == NULL ) {
    cleanup_po(1,aocoord_names);
    return PyErr_Format(PyExc_MemoryError,ex2err[EX2ERR_Allocation_Error],"");
  }

  ptr_sobj = (PyObject **)PyArray_DATA(aocoord_names);
  for (i=0; i<llen; i++) {
    coord_names[i] = PyString_AsString( ptr_sobj[i] );
  }

  rval = ex_put_coord_names(exoid,coord_names);

  free(coord_names);
  cleanup_po(1,aocoord_names);
  if ( rval < 0 )
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Library_Error],rval);

  return Py_BuildValue("i",rval);
}


//***************************************************************
// ex_get_node_num_map
//
static PyObject *ex2lib_ex_get_node_num_map(PyObject *self, PyObject *args) {
  int exoid, rval, num_nodes, comp_ws, io_ws;
  npy_intp npy_num_nodes;
  float fdmy;
  char cdmy;
  PyArrayObject *node_map;
  int *ptr_node_map;

  if (!PyArg_ParseTuple(args,
                        "(iii):ex_get_node_num_map",
                        &exoid,&comp_ws,&io_ws)
      )
    return PyErr_Format(PyExc_TypeError,ex2err[EX2ERR_Invalid_Argument],"");

  rval = ex_inquire(exoid,EX_INQ_NODES,&num_nodes,&fdmy,&cdmy);
  if ( rval < 0 )
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Inquire_Failed],"");

  npy_num_nodes = num_nodes;

  node_map = (PyArrayObject *)PyArray_EMPTY(1,&npy_num_nodes,PyArray_INT,
                                            PyArray_CORDER);
  if ( node_map == NULL )
    return PyErr_Format(PyExc_MemoryError,ex2err[EX2ERR_Allocation_Error],"");

  ptr_node_map = (int *)PyArray_DATA(node_map);

  rval = ex_get_node_num_map(exoid,ptr_node_map);
  if ( rval < 0 ) {
    cleanup_po(1,node_map);
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Library_Error],rval);
  }

  return PyArray_Return(node_map);
}


//***************************************************************
// ex_put_node_num_map
//
static PyObject *ex2lib_ex_put_node_num_map(PyObject *self, PyObject *args) {
  int exoid, rval, comp_ws, io_ws;
  PyObject *onode_map;
  PyArrayObject *node_map;
  int *ptr_node_map;

  if (!PyArg_ParseTuple(args,
                        "(iii)O:ex_put_node_num_map",
                        &exoid, &comp_ws, &io_ws, &onode_map)
      )
    return PyErr_Format(PyExc_TypeError,ex2err[EX2ERR_Invalid_Argument],"");

  node_map = (PyArrayObject *)intarray(onode_map,1,1);
  if ( node_map == NULL )
    return PyErr_Format(PyExc_MemoryError,ex2err[EX2ERR_Allocation_Error],"");

  ptr_node_map = (int *)PyArray_DATA(node_map);

  rval = ex_put_node_num_map(exoid,ptr_node_map);

  cleanup_po(1,node_map);
  if ( rval < 0 )
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Library_Error],rval);

  return Py_BuildValue("i",rval);
}


//***************************************************************
// ex_get_elem_num_map
//
static PyObject *ex2lib_ex_get_elem_num_map(PyObject *self, PyObject *args) {
  int exoid, rval, num_elem, comp_ws, io_ws;
  npy_intp npy_num_elem;
  float fdmy;
  char cdmy;
  PyArrayObject *elem_map;
  int *ptr_elem_map;

  if (!PyArg_ParseTuple(args,
                        "(iii):ex_get_elem_num_map",
                        &exoid, &comp_ws, &io_ws)
      )
    return PyErr_Format(PyExc_TypeError,ex2err[EX2ERR_Invalid_Argument],"");

  rval = ex_inquire(exoid,EX_INQ_ELEM,&num_elem,&fdmy,&cdmy);
  if ( rval < 0 )
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Inquire_Failed],"");

  npy_num_elem = num_elem;

  elem_map = (PyArrayObject *)PyArray_EMPTY(1,&npy_num_elem,PyArray_INT,
                                            PyArray_CORDER);
  if ( elem_map == NULL )
    return PyErr_Format(PyExc_MemoryError,ex2err[EX2ERR_Allocation_Error],"");

  ptr_elem_map = (int *)PyArray_DATA(elem_map);

  rval = ex_get_elem_num_map(exoid,ptr_elem_map);
  if ( rval < 0 ) {
    cleanup_po(1,elem_map);
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Library_Error],rval);
  }

  return PyArray_Return(elem_map);
}


//***************************************************************
// ex_put_elem_num_map
//
static PyObject *ex2lib_ex_put_elem_num_map(PyObject *self, PyObject *args) {
  int exoid, rval, comp_ws, io_ws;
  PyObject *oelem_map;
  PyArrayObject *elem_map;
  int *ptr_elem_map;

  if (!PyArg_ParseTuple(args,
                        "(iii)O:ex_put_elem_num_map",
                        &exoid, &comp_ws, &io_ws, &oelem_map)
      )
    return PyErr_Format(PyExc_TypeError,ex2err[EX2ERR_Invalid_Argument],"");

  elem_map = (PyArrayObject *)intarray(oelem_map,1,1);
  if ( elem_map == NULL )
    return PyErr_Format(PyExc_MemoryError,ex2err[EX2ERR_Allocation_Error],"");

  ptr_elem_map = (int *)PyArray_DATA(elem_map);

  rval = ex_put_elem_num_map(exoid,ptr_elem_map);

  cleanup_po(1,elem_map);
  if ( rval < 0 )
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Library_Error],rval);

  return Py_BuildValue("i",rval);
}


//***************************************************************
// ex_get_map
//
static PyObject *ex2lib_ex_get_map(PyObject *self, PyObject *args) {
  int exoid, rval, num_elem, comp_ws, io_ws;
  npy_intp npy_num_elem;
  float fdmy;
  char cdmy;
  PyArrayObject *elem_map;
  int *ptr_elem_map;

  if (!PyArg_ParseTuple(args,
                        "(iii):ex_get_map",
                        &exoid, &comp_ws, &io_ws)
      )
    return PyErr_Format(PyExc_TypeError,ex2err[EX2ERR_Invalid_Argument],"");

  rval = ex_inquire(exoid,EX_INQ_ELEM,&num_elem,&fdmy,&cdmy);
  if ( rval < 0 )
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Inquire_Failed],"");

  npy_num_elem = num_elem;

  elem_map = (PyArrayObject *)PyArray_EMPTY(1,&npy_num_elem,PyArray_INT,
                                            PyArray_CORDER);

  if ( elem_map == NULL )
    return PyErr_Format(PyExc_MemoryError,ex2err[EX2ERR_Allocation_Error],"");

  ptr_elem_map = (int *)PyArray_DATA(elem_map);

  rval = ex_get_map(exoid,ptr_elem_map);
  if ( rval < 0 ) {
    cleanup_po(1,elem_map);
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Library_Error],rval);
  }

  return PyArray_Return(elem_map);
}


//***************************************************************
// ex_put_map
//
static PyObject *ex2lib_ex_put_map(PyObject *self, PyObject *args) {
  int exoid, rval, comp_ws, io_ws;
  PyObject *oelem_map;
  PyArrayObject *elem_map;
  int *ptr_elem_map;

  if (!PyArg_ParseTuple(args,
                        "(iii)O:ex_put_map",
                        &exoid, &comp_ws, &io_ws, &oelem_map)
      )
    return PyErr_Format(PyExc_TypeError,ex2err[EX2ERR_Invalid_Argument],"");

  elem_map = (PyArrayObject *)intarray(oelem_map,1,1);
  if ( elem_map == NULL )
    return PyErr_Format(PyExc_MemoryError,ex2err[EX2ERR_Allocation_Error],"");

  ptr_elem_map = (int *)PyArray_DATA(elem_map);

  rval = ex_put_map(exoid,ptr_elem_map);

  cleanup_po(1,elem_map);
  if ( rval < 0 )
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Library_Error],rval);

  return Py_BuildValue("i",rval);
}


//***************************************************************
// ex_get_elem_block
//
static PyObject *ex2lib_ex_get_elem_block(PyObject *self, PyObject *args) {
  int exoid, elem_blk_id, num_elem_this_blk, num_nodes_per_elem, num_attr, rval;
  int comp_ws, io_ws;
  char elem_type[MAX_STR_LENGTH+1];

  if (!PyArg_ParseTuple(args,
                        "(iii)i:ex_get_elem_block",
                        &exoid, &comp_ws, &io_ws, &elem_blk_id)
      )
    return PyErr_Format(PyExc_TypeError,ex2err[EX2ERR_Invalid_Argument],"");

  rval = ex_get_elem_block(exoid,elem_blk_id,elem_type,&num_elem_this_blk,
                           &num_nodes_per_elem,&num_attr);
  if ( rval != 0 )
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Library_Error],rval);

  return Py_BuildValue("siii",elem_type,num_elem_this_blk,num_nodes_per_elem,
                       num_attr);
}


//***************************************************************
// ex_put_elem_block
//
static PyObject *ex2lib_ex_put_elem_block(PyObject *self, PyObject *args) {
  int exoid, elem_blk_id, num_elem_this_blk, num_nodes_per_elem, num_attr, rval;
  int comp_ws, io_ws;
  char *elem_type;

  if (!PyArg_ParseTuple(args,
                        "(iii)isiii:ex_put_elem_block",
                        &exoid, &comp_ws, &io_ws, &elem_blk_id, &elem_type,
                        &num_elem_this_blk, &num_nodes_per_elem, &num_attr)
      )
    return PyErr_Format(PyExc_TypeError,ex2err[EX2ERR_Invalid_Argument],"");

  rval = ex_put_elem_block(exoid,elem_blk_id,elem_type,num_elem_this_blk,
                           num_nodes_per_elem,num_attr);
  if ( rval < 0 )
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Library_Error],rval);

  return Py_BuildValue("i",rval);
}


//***************************************************************
// ex_get_elem_blk_ids
//
static PyObject *ex2lib_ex_get_elem_blk_ids(PyObject *self, PyObject *args) {
  int exoid, rval, num_eblks, comp_ws, io_ws;
  npy_intp npy_num_eblks;
  float fdmy;
  char cdmy;
  PyArrayObject *eblk_ids;
  int *ptr_eblk_ids;

  if (!PyArg_ParseTuple(args,
                        "(iii):ex_get_elem_blk_ids",
                        &exoid, &comp_ws, &io_ws)
      )
    return PyErr_Format(PyExc_TypeError,ex2err[EX2ERR_Invalid_Argument],"");

  rval = ex_inquire(exoid,EX_INQ_ELEM_BLK,&num_eblks,&fdmy,&cdmy);
  if ( rval < 0 )
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Inquire_Failed],"");

  npy_num_eblks = num_eblks;

  eblk_ids = (PyArrayObject *)PyArray_EMPTY(1,&npy_num_eblks,PyArray_INT,
                                            PyArray_CORDER);

  if ( eblk_ids == NULL )
    return PyErr_Format(PyExc_MemoryError,ex2err[EX2ERR_Allocation_Error],"");

  ptr_eblk_ids = (int *)PyArray_DATA(eblk_ids);

  rval = ex_get_elem_blk_ids(exoid,ptr_eblk_ids);
  if ( rval < 0 ) {
    cleanup_po(1,eblk_ids);
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Library_Error],rval);
  }

  return PyArray_Return(eblk_ids);
}


//***************************************************************
// ex_get_elem_conn
//
static PyObject *ex2lib_ex_get_elem_conn(PyObject *self, PyObject *args) {
  int exoid, elem_blk_id, rval, num_elem_this_block, num_nodes_per_elem;
  int num_attr, comp_ws, io_ws;
  npy_intp sz;
  PyArrayObject *conn;
  int *ptr_conn;
  char elem_type[MAX_STR_LENGTH+1];

  if (!PyArg_ParseTuple(args,
                        "(iii)i:ex_get_elem_conn",
                        &exoid, &comp_ws, &io_ws, &elem_blk_id)
      )
    return PyErr_Format(PyExc_TypeError,ex2err[EX2ERR_Invalid_Argument],"");

  rval = ex_get_elem_block(exoid,elem_blk_id,elem_type,&num_elem_this_block,
                           &num_nodes_per_elem,&num_attr);
  if ( rval < 0 )
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Inquire_Failed],"");

  sz = num_elem_this_block*num_nodes_per_elem;

  conn = (PyArrayObject *)PyArray_EMPTY(1,&sz,PyArray_INT,PyArray_CORDER);
  if ( conn == NULL )
    return PyErr_Format(PyExc_MemoryError,ex2err[EX2ERR_Allocation_Error],"");

  ptr_conn = (int *)PyArray_DATA(conn);

  rval = ex_get_elem_conn(exoid,elem_blk_id,ptr_conn);
  if ( rval < 0 ) {
    cleanup_po(1,conn);
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Library_Error],rval);
  }

  return PyArray_Return(conn);
}


//***************************************************************
// ex_put_elem_conn
//
static PyObject *ex2lib_ex_put_elem_conn(PyObject *self, PyObject *args) {
  int exoid, elem_blk_id, rval, comp_ws, io_ws;
  PyObject *oconn;
  PyArrayObject *conn;
  int *ptr_conn;

  if (!PyArg_ParseTuple(args,
                        "(iii)iO:ex_put_elem_conn",
                        &exoid, &comp_ws, &io_ws, &elem_blk_id, &oconn)
      )
    return PyErr_Format(PyExc_TypeError,ex2err[EX2ERR_Invalid_Argument],"");

  conn = (PyArrayObject *)intarray(oconn,1,1);
  if ( conn == NULL )
    return PyErr_Format(PyExc_MemoryError,ex2err[EX2ERR_Allocation_Error],"");

  ptr_conn = (int *)PyArray_DATA(conn);

  rval = ex_put_elem_conn(exoid,elem_blk_id,ptr_conn);

  cleanup_po(1,conn);
  if ( rval < 0 )
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Library_Error],rval);

  return Py_BuildValue("i",rval);
}


//***************************************************************
// ex_get_elem_attr
//
static PyObject *ex2lib_ex_get_elem_attr(PyObject *self, PyObject *args) {
  int exoid, elem_blk_id, rval, num_elem_this_block, num_nodes_per_elem;
  int num_attr, comp_ws, io_ws, fdtype;
  npy_intp sz;
  char elem_type[MAX_STR_LENGTH+1];
  PyArrayObject *attr;
  void *ptr_attr;

  if (!PyArg_ParseTuple(args,
                        "(iii)i:ex_get_elem_attr",
                        &exoid, &comp_ws, &io_ws, &elem_blk_id)
      )
    return PyErr_Format(PyExc_TypeError,ex2err[EX2ERR_Invalid_Argument],"");

  rval = ex_get_elem_block(exoid,elem_blk_id,elem_type,&num_elem_this_block,
                           &num_nodes_per_elem,&num_attr);
  if ( rval < 0 )
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Inquire_Failed],"");

  sz = num_elem_this_block*num_attr;

  if ( comp_ws == 4 )
    fdtype = PyArray_FLOAT32;
  else
    fdtype = PyArray_FLOAT64;

  attr = (PyArrayObject *)PyArray_EMPTY(1,&sz,fdtype,PyArray_CORDER);
  if ( attr == NULL )
    return PyErr_Format(PyExc_MemoryError,ex2err[EX2ERR_Allocation_Error],"");

  ptr_attr = (void *)PyArray_DATA(attr);

  rval = ex_get_elem_attr(exoid,elem_blk_id,ptr_attr);
  if ( rval < 0 ) {
    cleanup_po(1,attr);
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Library_Error],rval);
  }

  return PyArray_Return(attr);
}


//***************************************************************
// ex_put_elem_attr
//
static PyObject *ex2lib_ex_put_elem_attr(PyObject *self, PyObject *args) {
  int exoid, elem_blk_id, rval, comp_ws, io_ws, fdtype;
  PyObject *oattr;
  PyArrayObject *attr;
  void *ptr_attr;

  if (!PyArg_ParseTuple(args,
                        "(iii)iO:ex_put_elem_attr",
                        &exoid, &comp_ws, &io_ws, &elem_blk_id, &oattr)
      )
    return PyErr_Format(PyExc_TypeError,ex2err[EX2ERR_Invalid_Argument],"");

  if ( comp_ws == 4 )
    fdtype = PyArray_FLOAT32;
  else
    fdtype = PyArray_FLOAT64;

  attr = (PyArrayObject *)fparray(oattr,fdtype,1,1);
  if ( attr == NULL )
    return PyErr_Format(PyExc_MemoryError,ex2err[EX2ERR_Allocation_Error],"");

  ptr_attr = (void *)PyArray_DATA(attr);

  rval = ex_put_elem_attr(exoid,elem_blk_id,ptr_attr);

  cleanup_po(1,attr);
  if ( rval < 0 )
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Library_Error],rval);

  return Py_BuildValue("i",rval);
}


//***************************************************************
// ex_get_node_set_param
//
static PyObject *ex2lib_ex_get_node_set_param(PyObject *self, PyObject *args) {
  int exoid, node_set_id, num_nodes_in_set, num_dist_in_set, rval, comp_ws;
  int io_ws;

  if (!PyArg_ParseTuple(args,
                        "(iii)i:ex_get_node_set_param",
                        &exoid, &comp_ws, &io_ws, &node_set_id)
      )
    return PyErr_Format(PyExc_TypeError,ex2err[EX2ERR_Invalid_Argument],"");

  rval = ex_get_node_set_param(exoid,node_set_id,&num_nodes_in_set,
                               &num_dist_in_set);

  if ( rval != 0 )
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Library_Error],rval);

  return Py_BuildValue("ii",num_nodes_in_set,num_dist_in_set);
}


//***************************************************************
// ex_put_node_set_param
//
static PyObject *ex2lib_ex_put_node_set_param(PyObject *self, PyObject *args) {
  int exoid, node_set_id, num_nodes_in_set, num_dist_in_set, rval, comp_ws;
  int io_ws;

  if (!PyArg_ParseTuple(args,
                        "(iii)iii:ex_put_node_set_param",
                        &exoid, &comp_ws, &io_ws, &node_set_id,
                        &num_nodes_in_set, &num_dist_in_set)
      )
    return PyErr_Format(PyExc_TypeError,ex2err[EX2ERR_Invalid_Argument],"");

  rval = ex_put_node_set_param(exoid,node_set_id,num_nodes_in_set,
                               num_dist_in_set);
  if ( rval < 0 )
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Library_Error],rval);

  return Py_BuildValue("i",rval);
}


//***************************************************************
// ex_get_node_set
//
static PyObject *ex2lib_ex_get_node_set(PyObject *self, PyObject *args) {
  int exoid, node_set_id, rval, comp_ws, io_ws, num_nodes_in_set;
  npy_intp npy_num_nodes_in_set;
  int num_dist_in_set;
  PyArrayObject *nodes;
  int *ptr_nodes;

  if (!PyArg_ParseTuple(args,
                        "(iii)i:ex_get_node_set",
                        &exoid, &comp_ws, &io_ws, &node_set_id)
      )
    return PyErr_Format(PyExc_TypeError,ex2err[EX2ERR_Invalid_Argument],"");

  rval = ex_get_node_set_param(exoid,node_set_id,&num_nodes_in_set,
                               &num_dist_in_set);

  if ( rval < 0 )
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Inquire_Failed],"");

  npy_num_nodes_in_set = num_nodes_in_set;

  nodes = (PyArrayObject *)PyArray_EMPTY(1,&npy_num_nodes_in_set,PyArray_INT,
                                         PyArray_CORDER);
  if ( nodes == NULL )
    return PyErr_Format(PyExc_MemoryError,ex2err[EX2ERR_Allocation_Error],"");

  ptr_nodes = (int *)PyArray_DATA(nodes);

  rval = ex_get_node_set(exoid,node_set_id,ptr_nodes);
  if ( rval < 0 ) {
    cleanup_po(1,nodes);
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Library_Error],rval);
  }

  return PyArray_Return(nodes);
}


//***************************************************************
// ex_put_node_set
//
static PyObject *ex2lib_ex_put_node_set(PyObject *self, PyObject *args) {
  int exoid, node_set_id, rval, comp_ws, io_ws;
  PyObject *onodes;
  PyArrayObject *nodes;
  int *ptr_nodes;

  if (!PyArg_ParseTuple(args,
                        "(iii)iO:ex_put_node_set",
                        &exoid, &comp_ws, &io_ws, &node_set_id, &onodes)
      )
    return PyErr_Format(PyExc_TypeError,ex2err[EX2ERR_Invalid_Argument],"");

  nodes = (PyArrayObject *)intarray(onodes,1,1);
  if ( nodes == NULL )
    return PyErr_Format(PyExc_MemoryError,ex2err[EX2ERR_Allocation_Error],"");

  ptr_nodes = (int *)PyArray_DATA(nodes);

  rval = ex_put_node_set(exoid,node_set_id,ptr_nodes);

  cleanup_po(1,nodes);
  if ( rval < 0 )
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Library_Error],rval);

  return Py_BuildValue("i",rval);
}


//***************************************************************
// ex_get_node_set_dist_fact
//
static PyObject *ex2lib_ex_get_node_set_dist_fact(PyObject *self, PyObject *args) {
  int exoid, node_set_id, rval, comp_ws, io_ws, num_nodes_in_set;
  int num_dist_in_set, fdtype;
  npy_intp npy_num_dist_in_set;
  PyArrayObject *df;
  void *ptr_df;

  if (!PyArg_ParseTuple(args,
                        "(iii)i:ex_get_node_set_dist_fact",
                        &exoid, &comp_ws, &io_ws, &node_set_id)
      )
    return PyErr_Format(PyExc_TypeError,ex2err[EX2ERR_Invalid_Argument],"");

  rval = ex_get_node_set_param(exoid,node_set_id,&num_nodes_in_set,
                               &num_dist_in_set);

  if ( rval < 0 )
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Inquire_Failed],"");

  if ( comp_ws == 4 )
    fdtype = PyArray_FLOAT32;
  else
    fdtype = PyArray_FLOAT64;

  npy_num_dist_in_set = num_dist_in_set;

  df = (PyArrayObject *)PyArray_EMPTY(1,&npy_num_dist_in_set,fdtype,
                                      PyArray_CORDER);
  if ( df == NULL )
    return PyErr_Format(PyExc_MemoryError,ex2err[EX2ERR_Allocation_Error],"");

  ptr_df = (void *)PyArray_DATA(df);

  rval = ex_get_node_set_dist_fact(exoid,node_set_id,ptr_df);
  if ( rval < 0 ) {
    cleanup_po(1,df);
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Library_Error],rval);
  }

  return PyArray_Return(df);
}


//***************************************************************
// ex_put_node_set_dist_fact
//
static PyObject *ex2lib_ex_put_node_set_dist_fact(PyObject *self, PyObject *args) {
  int exoid, node_set_id, rval, comp_ws, io_ws, fdtype;
  PyObject *odf;
  PyArrayObject *df;
  void *ptr_df;

  if (!PyArg_ParseTuple(args,
                        "(iii)iO:ex_put_node_set_dist_fact",
                        &exoid, &comp_ws, &io_ws, &node_set_id, &odf)
      )
    return PyErr_Format(PyExc_TypeError,ex2err[EX2ERR_Invalid_Argument],"");

  if ( comp_ws == 4 )
    fdtype = PyArray_FLOAT32;
  else
    fdtype = PyArray_FLOAT64;

  df = (PyArrayObject *)fparray(odf,fdtype,1,1);
  if ( df == NULL )
    return PyErr_Format(PyExc_MemoryError,ex2err[EX2ERR_Allocation_Error],"");

  ptr_df = (void *)PyArray_DATA(df);

  rval = ex_put_node_set_dist_fact(exoid,node_set_id,ptr_df);

  cleanup_po(1,df);
  if ( rval < 0 )
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Library_Error],rval);

  return Py_BuildValue("i",rval);
}


//***************************************************************
// ex_get_node_set_ids
//
static PyObject *ex2lib_ex_get_node_set_ids(PyObject *self, PyObject *args) {
  int exoid, rval, num_node_sets, comp_ws, io_ws;
  npy_intp npy_num_node_sets;
  float fdmy;
  char cdmy;
  PyArrayObject *nsids;
  int *ptr_nsids;

  if (!PyArg_ParseTuple(args,
                        "(iii):ex_get_node_set_ids",
                        &exoid, &comp_ws, &io_ws)
      )
    return PyErr_Format(PyExc_TypeError,ex2err[EX2ERR_Invalid_Argument],"");

  rval = ex_inquire(exoid,EX_INQ_NODE_SETS,&num_node_sets,&fdmy,&cdmy);
  if ( rval < 0 )
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Inquire_Failed],"");

  npy_num_node_sets = num_node_sets;

  nsids = (PyArrayObject *)PyArray_EMPTY(1,&npy_num_node_sets,PyArray_INT,
                                         PyArray_CORDER);

  if ( nsids == NULL )
    return PyErr_Format(PyExc_MemoryError,ex2err[EX2ERR_Allocation_Error],"");

  ptr_nsids = (int *)PyArray_DATA(nsids);

  rval = ex_get_node_set_ids(exoid,ptr_nsids);
  if ( rval < 0 ) {
    cleanup_po(1,nsids);
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Library_Error],rval);
  }

  return PyArray_Return(nsids);
}


//***************************************************************
// ex_get_concat_node_sets
//
static PyObject *ex2lib_ex_get_concat_node_sets(PyObject *self, PyObject *args) {
  int exoid, rval, comp_ws, io_ws, num_node_sets, num_ns_nodes, num_ns_df;
  npy_intp  npy_num_node_sets, npy_num_ns_nodes, npy_num_ns_df;
  int fdtype;
  float fdmy;
  char cdmy;
  PyObject *rlst;
  PyArrayObject *node_set_ids, *num_nodes_per_set, *num_dist_per_set;
  PyArrayObject *node_sets_node_index, *node_sets_dist_index;
  PyArrayObject *node_sets_node_list;
  PyArrayObject *df;
  int *ptr_node_set_ids, *ptr_num_nodes_per_set, *ptr_num_dist_per_set;
  int *ptr_node_sets_node_index, *ptr_node_sets_dist_index;
  int *ptr_node_sets_node_list;
  void *ptr_df;

  if (!PyArg_ParseTuple(args,
                        "(iii):ex_get_concat_node_sets",
                        &exoid, &comp_ws, &io_ws)
      )
    return PyErr_Format(PyExc_TypeError,ex2err[EX2ERR_Invalid_Argument],"");

  rval = ex_inquire(exoid,EX_INQ_NODE_SETS,&num_node_sets,&fdmy,&cdmy);
  if ( rval < 0 )
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Inquire_Failed],"");

  rval = ex_inquire(exoid,EX_INQ_NS_NODE_LEN,&num_ns_nodes,&fdmy,&cdmy);
  if ( rval < 0 )
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Inquire_Failed],"");

  rval = ex_inquire(exoid,EX_INQ_NS_DF_LEN,&num_ns_df,&fdmy,&cdmy);
  if ( rval < 0 )
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Inquire_Failed],"");

  npy_num_node_sets = num_node_sets;
  npy_num_ns_nodes  = num_ns_nodes;
  npy_num_ns_df     = num_ns_df;

  node_set_ids =
    (PyArrayObject *)PyArray_EMPTY(1,&npy_num_node_sets,PyArray_INT,
                                   PyArray_CORDER);
  num_nodes_per_set =
    (PyArrayObject *)PyArray_EMPTY(1,&npy_num_node_sets,PyArray_INT,
                                   PyArray_CORDER);
  num_dist_per_set =
    (PyArrayObject *)PyArray_EMPTY(1,&npy_num_node_sets,PyArray_INT,
                                   PyArray_CORDER);
  node_sets_node_index =
    (PyArrayObject *)PyArray_EMPTY(1,&npy_num_node_sets,PyArray_INT,
                                   PyArray_CORDER);
  node_sets_dist_index =
    (PyArrayObject *)PyArray_EMPTY(1,&npy_num_node_sets,PyArray_INT,
                                   PyArray_CORDER);
  node_sets_node_list =
    (PyArrayObject *)PyArray_EMPTY(1,&npy_num_ns_nodes,PyArray_INT,
                                   PyArray_CORDER);

  if ( comp_ws == 4 )
    fdtype = PyArray_FLOAT32;
  else
    fdtype = PyArray_FLOAT64;

  df = (PyArrayObject *)PyArray_EMPTY(1,&npy_num_ns_df,fdtype,PyArray_CORDER);

  if ( (df == NULL) || (node_set_ids == NULL) || (num_nodes_per_set == NULL)
       || (num_dist_per_set == NULL) || (node_sets_node_index == NULL)
       || (node_sets_dist_index == NULL) || (node_sets_node_list == NULL) )
    {
      cleanup_po(7,df,node_set_ids,num_nodes_per_set,num_dist_per_set,
                 node_sets_node_index,node_sets_dist_index,node_sets_node_list);
      return PyErr_Format(PyExc_MemoryError,ex2err[EX2ERR_Allocation_Error],"");
    }

  ptr_df = (void *)PyArray_DATA(df);
  ptr_node_set_ids = (int *)PyArray_DATA(node_set_ids);
  ptr_num_nodes_per_set = (int *)PyArray_DATA(num_nodes_per_set);
  ptr_num_dist_per_set = (int *)PyArray_DATA(num_dist_per_set);
  ptr_node_sets_node_index = (int *)PyArray_DATA(node_sets_node_index);
  ptr_node_sets_dist_index = (int *)PyArray_DATA(node_sets_dist_index);
  ptr_node_sets_node_list = (int *)PyArray_DATA(node_sets_node_list);

  rval = ex_get_concat_node_sets(exoid,
                                 ptr_node_set_ids,
                                 ptr_num_nodes_per_set,
                                 ptr_num_dist_per_set,
                                 ptr_node_sets_node_index,
                                 ptr_node_sets_dist_index,
                                 ptr_node_sets_node_list,
                                 ptr_df);

  if ( rval < 0 ) {
    cleanup_po(7,df,node_set_ids,num_nodes_per_set,num_dist_per_set,
               node_sets_node_index,node_sets_dist_index,node_sets_node_list);
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Library_Error],rval);
  }

  rlst = PyList_New(7);

  PyList_SetItem(rlst,0,PyArray_Return(node_set_ids));
  PyList_SetItem(rlst,1,PyArray_Return(num_nodes_per_set));
  PyList_SetItem(rlst,2,PyArray_Return(num_dist_per_set));
  PyList_SetItem(rlst,3,PyArray_Return(node_sets_node_index));
  PyList_SetItem(rlst,4,PyArray_Return(node_sets_dist_index));
  PyList_SetItem(rlst,5,PyArray_Return(node_sets_node_list));
  PyList_SetItem(rlst,6,PyArray_Return(df));

  return rlst;
}


//***************************************************************
// ex_put_concat_node_sets
//
static PyObject *ex2lib_ex_put_concat_node_sets(PyObject *self, PyObject *args) {
  int exoid, rval, comp_ws, io_ws, fdtype;
  PyObject *onode_set_ids, *onum_nodes_per_set, *onum_dist_per_set;
  PyObject *onode_sets_node_index, *onode_sets_dist_index;
  PyObject *onode_sets_node_list, *odf;
  PyArrayObject *node_set_ids, *num_nodes_per_set, *num_dist_per_set;
  PyArrayObject *node_sets_node_index, *node_sets_dist_index;
  PyArrayObject *node_sets_node_list;
  PyArrayObject *df;
  int *ptr_node_set_ids, *ptr_num_nodes_per_set, *ptr_num_dist_per_set;
  int *ptr_node_sets_node_index, *ptr_node_sets_dist_index;
  int *ptr_node_sets_node_list;
  void *ptr_df;

  if (!PyArg_ParseTuple(args,
                        "(iii)OOOOOOO:ex_put_concat_node_sets",
                        &exoid, &comp_ws, &io_ws, &onode_set_ids,
                        &onum_nodes_per_set, &onum_dist_per_set,
                        &onode_sets_node_index, &onode_sets_dist_index,
                        &onode_sets_node_list, &odf)
      )
    return PyErr_Format(PyExc_TypeError,ex2err[EX2ERR_Invalid_Argument],"");

  if ( comp_ws == 4 )
    fdtype = PyArray_FLOAT32;
  else
    fdtype = PyArray_FLOAT64;

  node_set_ids         = (PyArrayObject *)intarray(onode_set_ids,1,1);
  num_nodes_per_set    = (PyArrayObject *)intarray(onum_nodes_per_set,1,1);
  num_dist_per_set     = (PyArrayObject *)intarray(onum_dist_per_set,1,1);
  node_sets_node_index = (PyArrayObject *)intarray(onode_sets_node_index,1,1);
  node_sets_dist_index = (PyArrayObject *)intarray(onode_sets_dist_index,1,1);
  node_sets_node_list  = (PyArrayObject *)intarray(onode_sets_node_list,1,1);

  df = (PyArrayObject *)fparray(odf,fdtype,1,1);

  if ( (df == NULL) || (node_set_ids == NULL) || (num_nodes_per_set == NULL)
       || (num_dist_per_set == NULL) || (node_sets_node_index == NULL)
       || (node_sets_dist_index == NULL) || (node_sets_node_list == NULL) )
    {
      cleanup_po(7,df,node_set_ids,num_nodes_per_set,num_dist_per_set,
                 node_sets_node_index,node_sets_dist_index,node_sets_node_list);
      return PyErr_Format(PyExc_MemoryError,ex2err[EX2ERR_Allocation_Error],"");
    }

  ptr_df = (void *)PyArray_DATA(df);
  ptr_node_set_ids = (int *)PyArray_DATA(node_set_ids);
  ptr_num_nodes_per_set = (int *)PyArray_DATA(num_nodes_per_set);
  ptr_num_dist_per_set = (int *)PyArray_DATA(num_dist_per_set);
  ptr_node_sets_node_index = (int *)PyArray_DATA(node_sets_node_index);
  ptr_node_sets_dist_index = (int *)PyArray_DATA(node_sets_dist_index);
  ptr_node_sets_node_list = (int *)PyArray_DATA(node_sets_node_list);

  rval = ex_put_concat_node_sets(exoid,
                                 ptr_node_set_ids,
                                 ptr_num_nodes_per_set,
                                 ptr_num_dist_per_set,
                                 ptr_node_sets_node_index,
                                 ptr_node_sets_dist_index,
                                 ptr_node_sets_node_list,
                                 ptr_df);

  cleanup_po(7,df,node_set_ids,num_nodes_per_set,num_dist_per_set,
             node_sets_node_index,node_sets_dist_index,node_sets_node_list);
  if ( rval < 0 )
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Library_Error],rval);

  return Py_BuildValue("i",rval);
}


//***************************************************************
// ex_get_side_set_param
//
static PyObject *ex2lib_ex_get_side_set_param(PyObject *self, PyObject *args) {
  int exoid, side_set_id, num_side_in_set, num_dist_fact_in_set, rval;
  int comp_ws, io_ws;

  if (!PyArg_ParseTuple(args,
                        "(iii)i:ex_get_side_set_param",
                        &exoid, &comp_ws, &io_ws, &side_set_id)
      )
    return PyErr_Format(PyExc_TypeError,ex2err[EX2ERR_Invalid_Argument],"");

  rval = ex_get_side_set_param(exoid,side_set_id,&num_side_in_set,
                               &num_dist_fact_in_set);

  if ( rval != 0 )
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Library_Error],rval);

  return Py_BuildValue("ii",num_side_in_set,num_dist_fact_in_set);
}


//***************************************************************
// ex_put_side_set_param
//
static PyObject *ex2lib_ex_put_side_set_param(PyObject *self, PyObject *args) {
  int exoid, side_set_id, num_side_in_set, num_dist_fact_in_set, rval;
  int comp_ws, io_ws;

  if (!PyArg_ParseTuple(args,
                        "(iii)iii:ex_put_side_set_param",
                        &exoid, &comp_ws, &io_ws, &side_set_id,
                        &num_side_in_set, &num_dist_fact_in_set)
      )
    return PyErr_Format(PyExc_TypeError,ex2err[EX2ERR_Invalid_Argument],"");

  rval = ex_put_side_set_param(exoid,side_set_id,num_side_in_set,
                               num_dist_fact_in_set);
  if ( rval < 0 )
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Library_Error],rval);

  return Py_BuildValue("i",rval);
}


//***************************************************************
// ex_get_side_set
//
static PyObject *ex2lib_ex_get_side_set(PyObject *self, PyObject *args) {
  int exoid, side_set_id, rval, comp_ws, io_ws, num_side_in_set;
  npy_intp npy_num_side_in_set;
  int num_dist_fact_in_set;
  PyObject *rlst;
  PyArrayObject *side_set_elem_list, *side_set_side_list;
  int *ptr_side_set_elem_list, *ptr_side_set_side_list;

  if (!PyArg_ParseTuple(args,
                        "(iii)i:ex_get_side_set",
                        &exoid, &comp_ws, &io_ws, &side_set_id)
      )
    return PyErr_Format(PyExc_TypeError,ex2err[EX2ERR_Invalid_Argument],"");

  rval = ex_get_side_set_param(exoid,side_set_id,&num_side_in_set,
                               &num_dist_fact_in_set);
  if ( rval < 0 )
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Inquire_Failed],"");

  npy_num_side_in_set = num_side_in_set;

  side_set_elem_list =
    (PyArrayObject *)PyArray_EMPTY(1,&npy_num_side_in_set,PyArray_INT,
                                   PyArray_CORDER);
  side_set_side_list =
    (PyArrayObject *)PyArray_EMPTY(1,&npy_num_side_in_set,PyArray_INT,
                                   PyArray_CORDER);

  if ( (side_set_elem_list == NULL) || (side_set_side_list == NULL) ) {
    cleanup_po(2,side_set_elem_list,side_set_side_list);
    return PyErr_Format(PyExc_MemoryError,ex2err[EX2ERR_Allocation_Error],"");
  }

  ptr_side_set_elem_list = (int *)PyArray_DATA(side_set_elem_list);
  ptr_side_set_side_list = (int *)PyArray_DATA(side_set_side_list);

  rval = ex_get_side_set(exoid,side_set_id,ptr_side_set_elem_list,
                         ptr_side_set_side_list);

  if ( rval < 0 ) {
    cleanup_po(2,side_set_elem_list,side_set_side_list);
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Library_Error],rval);
  }

  rlst = PyList_New(2);
  PyList_SetItem(rlst,0,PyArray_Return(side_set_elem_list));
  PyList_SetItem(rlst,1,PyArray_Return(side_set_side_list));

  return rlst;
}


//***************************************************************
// ex_put_side_set
//
static PyObject *ex2lib_ex_put_side_set(PyObject *self, PyObject *args) {
  int exoid, side_set_id, rval, comp_ws, io_ws;
  PyObject *oside_set_elem_list, *oside_set_side_list;
  PyArrayObject *side_set_elem_list, *side_set_side_list;
  int *ptr_side_set_elem_list, *ptr_side_set_side_list;

  if (!PyArg_ParseTuple(args,
                        "(iii)iOO:ex_put_side_set",
                        &exoid, &comp_ws, &io_ws, &side_set_id,
                        &oside_set_elem_list, &oside_set_side_list)
      )
    return PyErr_Format(PyExc_TypeError,ex2err[EX2ERR_Invalid_Argument],"");

  side_set_elem_list = (PyArrayObject *)intarray(oside_set_elem_list,1,1);
  side_set_side_list = (PyArrayObject *)intarray(oside_set_side_list,1,1);

  if ( (side_set_elem_list == NULL) || (side_set_side_list == NULL) ) {
    cleanup_po(2,side_set_elem_list,side_set_side_list);
    return PyErr_Format(PyExc_MemoryError,ex2err[EX2ERR_Allocation_Error],"");
  }

  ptr_side_set_elem_list = (int *)PyArray_DATA(side_set_elem_list);
  ptr_side_set_side_list = (int *)PyArray_DATA(side_set_side_list);

  rval = ex_put_side_set(exoid,side_set_id,ptr_side_set_elem_list,
                         ptr_side_set_side_list);

  cleanup_po(2,side_set_elem_list,side_set_side_list);
  if ( rval < 0 )
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Library_Error],rval);

  return Py_BuildValue("i",rval);
}


//***************************************************************
// ex_get_side_set_dist_fact
//
static PyObject *ex2lib_ex_get_side_set_dist_fact(PyObject *self, PyObject *args) {
  int exoid, side_set_id, rval, comp_ws, io_ws, num_side_in_set;
  int num_dist_fact_in_set, fdtype;
  npy_intp npy_num_dist_fact_in_set;
  PyArrayObject *df;
  void *ptr_df;

  if (!PyArg_ParseTuple(args,
                        "(iii)i:ex_get_side_set_dist_fact",
                        &exoid, &comp_ws, &io_ws, &side_set_id)
      )
    return PyErr_Format(PyExc_TypeError,ex2err[EX2ERR_Invalid_Argument],"");

  rval = ex_get_side_set_param(exoid,side_set_id,&num_side_in_set,
                               &num_dist_fact_in_set);
  if ( rval < 0 )
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Inquire_Failed],"");

  if ( comp_ws == 4 )
    fdtype = PyArray_FLOAT32;
  else
    fdtype = PyArray_FLOAT64;

  npy_num_dist_fact_in_set = num_dist_fact_in_set;

  df = (PyArrayObject *)PyArray_EMPTY(1,&npy_num_dist_fact_in_set,fdtype,
                                      PyArray_CORDER);
  if ( df == NULL )
    return PyErr_Format(PyExc_MemoryError,ex2err[EX2ERR_Allocation_Error],"");

  ptr_df = (void *)PyArray_DATA(df);

  rval = ex_get_side_set_dist_fact(exoid,side_set_id,ptr_df);

  if ( rval < 0 ) {
    cleanup_po(1,df);
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Library_Error],rval);
  }

  return PyArray_Return(df);
}


//***************************************************************
// ex_put_side_set_dist_fact
//
static PyObject *ex2lib_ex_put_side_set_dist_fact(PyObject *self, PyObject *args) {
  int exoid, side_set_id, rval, comp_ws, io_ws, fdtype;
  PyObject *odf;
  PyArrayObject *df;
  void *ptr_df;

  if (!PyArg_ParseTuple(args,
                        "(iii)iO:ex_put_side_set_dist_fact",
                        &exoid, &comp_ws, &io_ws, &side_set_id, &odf)
      )
    return PyErr_Format(PyExc_TypeError,ex2err[EX2ERR_Invalid_Argument],"");

  if ( comp_ws == 4 )
    fdtype = PyArray_FLOAT32;
  else
    fdtype = PyArray_FLOAT64;

  df = (PyArrayObject *)fparray(odf,fdtype,1,1);
  if ( df == NULL )
    return PyErr_Format(PyExc_MemoryError,ex2err[EX2ERR_Allocation_Error],"");

  ptr_df = (void *)PyArray_DATA(df);

  rval = ex_put_side_set_dist_fact(exoid,side_set_id,ptr_df);

  cleanup_po(1,df);
  if ( rval < 0 )
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Library_Error],rval);

  return Py_BuildValue("i",rval);
}


//***************************************************************
// ex_get_concat_side_sets
//
static PyObject *ex2lib_ex_get_concat_side_sets(PyObject *self, PyObject *args) {
  int exoid, rval, comp_ws, io_ws, num_ss, num_ss_elem, num_ss_df, fdtype;
  npy_intp npy_num_ss, npy_num_ss_elem, npy_num_ss_df;
  float fdmy;
  char cdmy;
  PyObject *rlst;
  PyArrayObject *side_set_ids, *num_side_per_set, *num_dist_per_set;
  PyArrayObject *side_sets_elem_index, *side_sets_dist_index;
  PyArrayObject *side_sets_elem_list, *side_sets_side_list;
  PyArrayObject *df;
  int *ptr_side_set_ids, *ptr_num_side_per_set, *ptr_num_dist_per_set;
  int *ptr_side_sets_elem_index, *ptr_side_sets_dist_index;
  int *ptr_side_sets_elem_list, *ptr_side_sets_side_list;
  void *ptr_df;

  if (!PyArg_ParseTuple(args,
                        "(iii):ex_get_concat_side_sets",
                        &exoid, &comp_ws, &io_ws)
      )
    return PyErr_Format(PyExc_TypeError,ex2err[EX2ERR_Invalid_Argument],"");

  rval = ex_inquire(exoid,EX_INQ_SIDE_SETS,&num_ss,&fdmy,&cdmy);
  if ( rval < 0 )
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Inquire_Failed],"");

  rval = ex_inquire(exoid,EX_INQ_SS_ELEM_LEN,&num_ss_elem,&fdmy,&cdmy);
  if ( rval < 0 )
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Inquire_Failed],"");

  rval = ex_inquire(exoid,EX_INQ_SS_DF_LEN,&num_ss_df,&fdmy,&cdmy);
  if ( rval < 0 )
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Inquire_Failed],"");

  if ( comp_ws == 4 )
    fdtype = PyArray_FLOAT32;
  else
    fdtype = PyArray_FLOAT64;

  npy_num_ss      = num_ss;
  npy_num_ss_elem = num_ss_elem;
  npy_num_ss_df   = num_ss_df;

  side_set_ids =
    (PyArrayObject *)PyArray_EMPTY(1,&npy_num_ss,PyArray_INT,PyArray_CORDER);
  num_side_per_set =
    (PyArrayObject *)PyArray_EMPTY(1,&npy_num_ss,PyArray_INT,PyArray_CORDER);
  num_dist_per_set =
    (PyArrayObject *)PyArray_EMPTY(1,&npy_num_ss,PyArray_INT,PyArray_CORDER);
  side_sets_elem_index =
    (PyArrayObject *)PyArray_EMPTY(1,&npy_num_ss,PyArray_INT,PyArray_CORDER);
  side_sets_dist_index =
    (PyArrayObject *)PyArray_EMPTY(1,&npy_num_ss,PyArray_INT,PyArray_CORDER);
  side_sets_elem_list =
    (PyArrayObject *)PyArray_EMPTY(1,&npy_num_ss_elem,PyArray_INT,PyArray_CORDER);
  side_sets_side_list =
    (PyArrayObject *)PyArray_EMPTY(1,&npy_num_ss_elem,PyArray_INT,PyArray_CORDER);

  df = (PyArrayObject *)PyArray_EMPTY(1,&npy_num_ss_df,fdtype,PyArray_CORDER);

  if ( (df == NULL) || (side_set_ids == NULL) || (num_side_per_set == NULL)
       || (num_dist_per_set == NULL) || (side_sets_elem_index == NULL)
       || (side_sets_dist_index == NULL) || (side_sets_elem_list == NULL)
       || (side_sets_side_list == NULL) )
    {
      cleanup_po(8,df,side_set_ids,num_side_per_set,num_dist_per_set,
                 side_sets_elem_index,side_sets_dist_index,side_sets_elem_list,
                 side_sets_side_list);
      return PyErr_Format(PyExc_MemoryError,ex2err[EX2ERR_Allocation_Error],"");
    }

  ptr_df = (void *)PyArray_DATA(df);
  ptr_side_set_ids = (int *)PyArray_DATA(side_set_ids);
  ptr_num_side_per_set = (int *)PyArray_DATA(num_side_per_set);
  ptr_num_dist_per_set = (int *)PyArray_DATA(num_dist_per_set);
  ptr_side_sets_elem_index = (int *)PyArray_DATA(side_sets_elem_index);
  ptr_side_sets_dist_index = (int *)PyArray_DATA(side_sets_dist_index);
  ptr_side_sets_elem_list = (int *)PyArray_DATA(side_sets_elem_list);
  ptr_side_sets_side_list = (int *)PyArray_DATA(side_sets_side_list);

  rval = ex_get_concat_side_sets(exoid,
                                 ptr_side_set_ids,
                                 ptr_num_side_per_set,
                                 ptr_num_dist_per_set,
                                 ptr_side_sets_elem_index,
                                 ptr_side_sets_dist_index,
                                 ptr_side_sets_elem_list,
                                 ptr_side_sets_side_list,
                                 ptr_df);

  if ( rval < 0 ) {
    cleanup_po(8,df,side_set_ids,num_side_per_set,num_dist_per_set,
               side_sets_elem_index,side_sets_dist_index,side_sets_elem_list,
               side_sets_side_list);
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Library_Error],rval);
  }

  rlst = PyList_New(8);

  PyList_SetItem(rlst,0,PyArray_Return(side_set_ids));
  PyList_SetItem(rlst,1,PyArray_Return(num_side_per_set));
  PyList_SetItem(rlst,2,PyArray_Return(num_dist_per_set));
  PyList_SetItem(rlst,3,PyArray_Return(side_sets_elem_index));
  PyList_SetItem(rlst,4,PyArray_Return(side_sets_dist_index));
  PyList_SetItem(rlst,5,PyArray_Return(side_sets_elem_list));
  PyList_SetItem(rlst,6,PyArray_Return(side_sets_side_list));
  PyList_SetItem(rlst,7,PyArray_Return(df));

  return rlst;
}


//***************************************************************
// ex_put_concat_side_sets
//
static PyObject *ex2lib_ex_put_concat_side_sets(PyObject *self, PyObject *args) {
  int exoid, rval, comp_ws, io_ws, fdtype;
  PyObject *oside_set_ids, *onum_side_per_set, *onum_dist_per_set;
  PyObject *oside_sets_elem_index, *oside_sets_dist_index;
  PyObject *oside_sets_elem_list, *oside_sets_side_list, *odf;
  PyArrayObject *side_set_ids, *num_side_per_set, *num_dist_per_set;
  PyArrayObject *side_sets_elem_index, *side_sets_dist_index;
  PyArrayObject *side_sets_elem_list, *side_sets_side_list;
  PyArrayObject *df;
  int *ptr_side_set_ids, *ptr_num_side_per_set, *ptr_num_dist_per_set;
  int *ptr_side_sets_elem_index, *ptr_side_sets_dist_index;
  int *ptr_side_sets_elem_list, *ptr_side_sets_side_list;
  void *ptr_df;

  if (!PyArg_ParseTuple(args,
                        "(iii)OOOOOOO:ex_put_concat_side_sets",
                        &exoid, &comp_ws, &io_ws, &oside_set_ids,
                        &onum_side_per_set, &onum_dist_per_set,
                        &oside_sets_elem_index, &oside_sets_dist_index,
                        &oside_sets_elem_list, &oside_sets_side_list, &odf)
      )
    return PyErr_Format(PyExc_TypeError,ex2err[EX2ERR_Invalid_Argument],"");

  if ( comp_ws == 4 )
    fdtype = PyArray_FLOAT32;
  else
    fdtype = PyArray_FLOAT64;

  side_set_ids         = (PyArrayObject *)intarray(oside_set_ids,1,1);
  num_side_per_set     = (PyArrayObject *)intarray(onum_side_per_set,1,1);
  num_dist_per_set     = (PyArrayObject *)intarray(onum_dist_per_set,1,1);
  side_sets_elem_index = (PyArrayObject *)intarray(oside_sets_elem_index,1,1);
  side_sets_dist_index = (PyArrayObject *)intarray(oside_sets_dist_index,1,1);
  side_sets_elem_list  = (PyArrayObject *)intarray(oside_sets_elem_list,1,1);
  side_sets_side_list  = (PyArrayObject *)intarray(oside_sets_side_list,1,1);

  df = (PyArrayObject *)fparray(odf,fdtype,1,1);

  if ( (df == NULL) || (side_set_ids == NULL) || (num_side_per_set == NULL)
       || (num_dist_per_set == NULL) || (side_sets_elem_index == NULL)
       || (side_sets_dist_index == NULL) || (side_sets_elem_list == NULL)
       || (side_sets_side_list == NULL) )
    {
      cleanup_po(8,df,side_set_ids,num_side_per_set,num_dist_per_set,
                 side_sets_elem_index,side_sets_dist_index,side_sets_elem_list,
                 side_sets_side_list);
      return PyErr_Format(PyExc_MemoryError,ex2err[EX2ERR_Allocation_Error],"");
    }

  ptr_df = (void *)PyArray_DATA(df);
  ptr_side_set_ids = (int *)PyArray_DATA(side_set_ids);
  ptr_num_side_per_set = (int *)PyArray_DATA(num_side_per_set);
  ptr_num_dist_per_set = (int *)PyArray_DATA(num_dist_per_set);
  ptr_side_sets_elem_index = (int *)PyArray_DATA(side_sets_elem_index);
  ptr_side_sets_dist_index = (int *)PyArray_DATA(side_sets_dist_index);
  ptr_side_sets_elem_list = (int *)PyArray_DATA(side_sets_elem_list);
  ptr_side_sets_side_list = (int *)PyArray_DATA(side_sets_side_list);

  rval = ex_put_concat_side_sets(exoid,
                                 ptr_side_set_ids,
                                 ptr_num_side_per_set,
                                 ptr_num_dist_per_set,
                                 ptr_side_sets_elem_index,
                                 ptr_side_sets_dist_index,
                                 ptr_side_sets_elem_list,
                                 ptr_side_sets_side_list,
                                 ptr_df);

  cleanup_po(8,df,side_set_ids,num_side_per_set,num_dist_per_set,
             side_sets_elem_index,side_sets_dist_index,side_sets_elem_list,
             side_sets_side_list);
  if ( rval < 0 )
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Library_Error],rval);

  return Py_BuildValue("i",rval);
}


//***************************************************************
// ex_get_prop_names
//
static PyObject *ex2lib_ex_get_prop_names(PyObject *self, PyObject *args) {
  int exoid, obj_type, num_props, rval, i, comp_ws, io_ws;
  float fdmy;
  char cdmy;
  PyObject *opname;
  char *pnames, **pname;

  if (!PyArg_ParseTuple(args,
                        "(iii)i:ex_get_prop_names",
                        &exoid, &comp_ws, &io_ws, &obj_type)
      )
    return PyErr_Format(PyExc_TypeError,ex2err[EX2ERR_Invalid_Argument],"");

  if ( obj_type == EX_ELEM_BLOCK )
    rval = ex_inquire(exoid,EX_INQ_EB_PROP,&num_props,&fdmy,&cdmy);
  else if ( obj_type == EX_NODE_SET )
    rval = ex_inquire(exoid,EX_INQ_NS_PROP,&num_props,&fdmy,&cdmy);
  else if ( obj_type == EX_SIDE_SET )
    rval = ex_inquire(exoid,EX_INQ_SS_PROP,&num_props,&fdmy,&cdmy);
  else
    return PyErr_Format(PyExc_TypeError,ex2err[EX2ERR_Invalid_Argument],"");

  if ( rval < 0 )
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Inquire_Failed],"");

  pnames = (char *)malloc(num_props*(MAX_STR_LENGTH+1)*sizeof(char));
  pname  = (char **)malloc(num_props*sizeof(char *));
  if ( ( pnames == NULL ) || ( pname == NULL ) ) {
    return PyErr_Format(PyExc_MemoryError,ex2err[EX2ERR_Allocation_Error],"");
  }

  for ( i = 0; i < num_props; i++ ) {
      pname[i] = pnames +i*(MAX_STR_LENGTH+1);
  }

  rval = ex_get_prop_names(exoid,obj_type,pname);

  if ( rval < 0 ) {
    free(pname);
    free(pnames);
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Library_Error],rval);
  }

  opname = PyList_New(num_props);
  for ( i = 0; i < num_props; i++ )
    PyList_SetItem(opname,i,Py_BuildValue("s",pname[i]));

  free(pname);
  free(pnames);
  return opname;
}


//***************************************************************
// ex_put_prop_names
//
static PyObject *ex2lib_ex_put_prop_names(PyObject *self, PyObject *args) {
  int exoid, obj_type, num_props, rval, i, comp_ws, io_ws;
  PyObject *oprop_names;
  PyArrayObject *aoprop_names;
  PyObject **ptr_sobj;
  char **prop_names;

  if (!PyArg_ParseTuple(args,
                        "(iii)iiO:ex_put_prop_names",
                        &exoid, &comp_ws, &io_ws, &obj_type, &num_props,
                        &oprop_names)
      )
    return PyErr_Format(PyExc_TypeError,ex2err[EX2ERR_Invalid_Argument],"");

  aoprop_names =
    (PyArrayObject *)PyArray_ContiguousFromAny(oprop_names,
                                               PyArray_OBJECT,1,1);
  if ( aoprop_names == NULL )
    return PyErr_Format(PyExc_MemoryError,ex2err[EX2ERR_Allocation_Error],"");

  prop_names = (char **)malloc(num_props*sizeof(char *));
  if ( prop_names == NULL ) {
    cleanup_po(1,aoprop_names);
    return PyErr_Format(PyExc_MemoryError,ex2err[EX2ERR_Allocation_Error],"");
  }

  ptr_sobj = (PyObject **)PyArray_DATA(aoprop_names);
  for (i=0; i<num_props; i++) {
    prop_names[i] = PyString_AsString( ptr_sobj[i] );
  }

  rval = ex_put_prop_names(exoid,obj_type,num_props,prop_names);

  free(prop_names);
  cleanup_po(1,aoprop_names);
  if ( rval < 0 )
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Library_Error],rval);

  return Py_BuildValue("i",rval);
}


//***************************************************************
// ex_get_prop
//
static PyObject *ex2lib_ex_get_prop(PyObject *self, PyObject *args) {
  int exoid, obj_type, obj_id, value, rval, comp_ws, io_ws;
  char *prop_name;

  if (!PyArg_ParseTuple(args,
                        "(iii)iis:ex_get_prop",
                        &exoid, &comp_ws, &io_ws, &obj_type, &obj_id,
                        &prop_name)
      )
    return PyErr_Format(PyExc_TypeError,ex2err[EX2ERR_Invalid_Argument],"");

  rval = ex_get_prop(exoid,obj_type,obj_id,prop_name,&value);
  // Fail if not zero.
  if ( rval != 0 )
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Library_Error],rval);

  return Py_BuildValue("i",value);
}


//***************************************************************
// ex_put_prop
//
static PyObject *ex2lib_ex_put_prop(PyObject *self, PyObject *args) {
  int exoid, obj_type, obj_id, value, rval, comp_ws, io_ws;
  char *prop_name;

  if (!PyArg_ParseTuple(args,
                        "(iii)iisi:ex_put_prop",
                        &exoid, &comp_ws, &io_ws, &obj_type, &obj_id,
                        &prop_name, &value)
      )
    return PyErr_Format(PyExc_TypeError,ex2err[EX2ERR_Invalid_Argument],"");

  rval = ex_put_prop(exoid,obj_type,obj_id,prop_name,value);
  if ( rval < 0 )
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Library_Error],rval);

  return Py_BuildValue("i",rval);
}


//***************************************************************
// ex_get_prop_array
//
static PyObject *ex2lib_ex_get_prop_array(PyObject *self, PyObject *args) {
  int exoid, obj_type, rval, comp_ws, io_ws, count;
  npy_intp npy_count;
  float fdmy;
  char cdmy;
  char *prop_name;
  PyArrayObject *value;
  int *ptr_value;

  if (!PyArg_ParseTuple(args,
                        "(iii)is:ex_get_prop_array",
                        &exoid, &comp_ws, &io_ws, &obj_type, &prop_name)
      )
    return PyErr_Format(PyExc_TypeError,ex2err[EX2ERR_Invalid_Argument],"");

  if ( obj_type == EX_ELEM_BLOCK )
    rval = ex_inquire(exoid,EX_INQ_ELEM_BLK,&count,&fdmy,&cdmy);
  else if ( obj_type == EX_NODE_SET )
    rval = ex_inquire(exoid,EX_INQ_NODE_SETS,&count,&fdmy,&cdmy);
  else if ( obj_type == EX_SIDE_SET )
    rval = ex_inquire(exoid,EX_INQ_SIDE_SETS,&count,&fdmy,&cdmy);
  else
    return PyErr_Format(PyExc_TypeError,ex2err[EX2ERR_Invalid_Argument],"");

  if ( rval < 0 )
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Inquire_Failed],"");

  npy_count = count;

  value = (PyArrayObject *)PyArray_EMPTY(1,&npy_count,PyArray_INT,PyArray_CORDER);
  if ( value == NULL )
    return PyErr_Format(PyExc_MemoryError,ex2err[EX2ERR_Allocation_Error],"");

  ptr_value = (int *)PyArray_DATA(value);

  rval = ex_get_prop_array(exoid,obj_type,prop_name,ptr_value);

  if ( rval < 0 ) {
    cleanup_po(1,value);
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Library_Error],rval);
  }

  return PyArray_Return(value);
}


//***************************************************************
// ex_put_prop_array
//
static PyObject *ex2lib_ex_put_prop_array(PyObject *self, PyObject *args) {
  int exoid, obj_type, rval, comp_ws, io_ws;
  char *prop_name;
  PyObject *ovalue;
  PyArrayObject *value;
  int *ptr_value;

  if (!PyArg_ParseTuple(args,
                        "(iii)isO:ex_put_prop_array",
                        &exoid, &comp_ws, &io_ws, &obj_type, &prop_name,
                        &ovalue)
      )
    return PyErr_Format(PyExc_TypeError,ex2err[EX2ERR_Invalid_Argument],"");

  value = (PyArrayObject *)intarray(ovalue,1,1);
  if ( value == NULL )
    return PyErr_Format(PyExc_MemoryError,ex2err[EX2ERR_Allocation_Error],"");

  ptr_value = (int *)PyArray_DATA(value);

  rval = ex_put_prop_array(exoid,obj_type,prop_name,ptr_value);

  cleanup_po(1,value);
  if ( rval < 0 )
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Library_Error],rval);

  return Py_BuildValue("i",rval);
}


//***************************************************************
// ex_get_var_param
//
static PyObject *ex2lib_ex_get_var_param(PyObject *self, PyObject *args) {
  int exoid, num_vars, rval, comp_ws, io_ws;
  char *var_type;

  if (!PyArg_ParseTuple(args,
                        "(iii)s:ex_get_var_param",
                        &exoid, &comp_ws, &io_ws, &var_type)
      )
    return PyErr_Format(PyExc_TypeError,ex2err[EX2ERR_Invalid_Argument],"");

  rval = ex_get_var_param(exoid,var_type,&num_vars);
  if ( rval != 0 )
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Library_Error],rval);

  return Py_BuildValue("i",num_vars);
}


//***************************************************************
// ex_put_var_param
//
static PyObject *ex2lib_ex_put_var_param(PyObject *self, PyObject *args) {
  int exoid, num_vars, rval, comp_ws, io_ws;
  char *var_type;

  if (!PyArg_ParseTuple(args,
                        "(iii)si:ex_put_var_param",
                        &exoid, &comp_ws, &io_ws, &var_type, &num_vars)
      )
    return PyErr_Format(PyExc_TypeError,ex2err[EX2ERR_Invalid_Argument],"");

  rval = ex_put_var_param(exoid,var_type,num_vars);
  if ( rval < 0 )
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Library_Error],rval);

  return Py_BuildValue("i",rval);
}


//***************************************************************
// ex_get_var_names
//
static PyObject *ex2lib_ex_get_var_names(PyObject *self, PyObject *args) {
  int i, exoid, num_vars, rval, comp_ws, io_ws;
  char *var_type;
  PyObject *oname;
  char *names, **name;

  if (!PyArg_ParseTuple(args,
                        "(iii)s:ex_get_var_names",
                        &exoid, &comp_ws, &io_ws, &var_type)
      )
    return PyErr_Format(PyExc_TypeError,ex2err[EX2ERR_Invalid_Argument],"");

  rval = ex_get_var_param(exoid,var_type,&num_vars);
  if ( rval < 0 )
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Inquire_Failed],"");

  names = (char *)malloc(num_vars*(MAX_STR_LENGTH+1)*sizeof(char));
  name  = (char **)malloc(num_vars*sizeof(char *));
  if ( ( names == NULL ) || ( name == NULL ) ) {
    return PyErr_Format(PyExc_MemoryError,ex2err[EX2ERR_Allocation_Error],"");
  }

  for ( i = 0; i < num_vars; i++ ) {
      name[i] = names +i*(MAX_STR_LENGTH+1);
  }

  rval = ex_get_var_names(exoid,var_type,num_vars,name);
  if ( rval < 0 ) {
    free(name);
    free(names);
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Library_Error],rval);
  }

  oname = PyList_New(num_vars);
  for ( i = 0; i < num_vars; i++ )
    PyList_SetItem(oname,i,Py_BuildValue("s",name[i]));

  free(name);
  free(names);
  return oname;
}


//***************************************************************
// ex_put_var_names
//
static PyObject *ex2lib_ex_put_var_names(PyObject *self, PyObject *args) {
  int i, exoid, num_vars, rval, comp_ws, io_ws;
  char *var_type;
  PyObject *onames;
  PyArrayObject *aonames;
  PyObject **ptr_sobj;
  char **names;

  if (!PyArg_ParseTuple(args,
                        "(iii)siO:ex_put_var_names",
                        &exoid, &comp_ws, &io_ws, &var_type, &num_vars, &onames)
      )
    return PyErr_Format(PyExc_TypeError,ex2err[EX2ERR_Invalid_Argument],"");

  aonames = (PyArrayObject *)PyArray_ContiguousFromAny(onames,
                                                       PyArray_OBJECT,1,1);
  if ( aonames == NULL )
    return PyErr_Format(PyExc_MemoryError,ex2err[EX2ERR_Allocation_Error],"");

  names = (char **)malloc(num_vars*sizeof(char *));
  if ( names == NULL ) {
    cleanup_po(1,aonames);
    return PyErr_Format(PyExc_MemoryError,ex2err[EX2ERR_Allocation_Error],"");
  }

  ptr_sobj = (PyObject **)PyArray_DATA(aonames);
  for (i=0; i<num_vars; i++) {
    names[i] = PyString_AsString( ptr_sobj[i] );
  }

  rval = ex_put_var_names(exoid,var_type,num_vars,names);

  free(names);
  cleanup_po(1,aonames);
  if ( rval < 0 )
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Library_Error],rval);

  return Py_BuildValue("i",rval);
}


//***************************************************************
// ex_get_time
//
static PyObject *ex2lib_ex_get_time(PyObject *self, PyObject *args) {
  int exoid, time_step, rval, comp_ws, io_ws;
  npy_float32 stime_value;
  npy_float64 dtime_value;

  if (!PyArg_ParseTuple(args,
                        "(iii)i:ex_get_time",
                        &exoid, &comp_ws, &io_ws, &time_step)
      )
    return PyErr_Format(PyExc_TypeError,ex2err[EX2ERR_Invalid_Argument],"");

  if ( comp_ws == 4 ) {
    rval = ex_get_time(exoid,time_step,&stime_value);
    dtime_value = stime_value;
  }
  else
    rval = ex_get_time(exoid,time_step,&dtime_value);

  if ( rval < 0 )
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Library_Error],rval);

  return Py_BuildValue("d",dtime_value);
}


//***************************************************************
// ex_put_time
//
static PyObject *ex2lib_ex_put_time(PyObject *self, PyObject *args) {
  int exoid, time_step, rval, comp_ws, io_ws;
  npy_float32 stime_value;
  npy_float64 dtime_value;

  if (!PyArg_ParseTuple(args,
                        "(iii)id:ex_put_time",
                        &exoid, &comp_ws, &io_ws, &time_step, &dtime_value)
      )
    return PyErr_Format(PyExc_TypeError,ex2err[EX2ERR_Invalid_Argument],"");

  stime_value = (npy_float32)dtime_value;
  if ( comp_ws == 4 )
    rval = ex_put_time(exoid,time_step,&stime_value);
  else
    rval = ex_put_time(exoid,time_step,&dtime_value);

  if ( rval < 0 )
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Library_Error],rval);

  return Py_BuildValue("i",rval);
}


//***************************************************************
// ex_get_elem_var_tab
//
static PyObject *ex2lib_ex_get_elem_var_tab(PyObject *self, PyObject *args) {
  int exoid, num_elem_blk, num_elem_var, rval, comp_ws, io_ws;
  npy_intp sz;
  float fdmy;
  char cdmy;
  PyArrayObject *tab;
  int *ptr_tab;

  if (!PyArg_ParseTuple(args,
                        "(iii):ex_get_elem_var_tab",
                        &exoid, &comp_ws, &io_ws)
      )
    return PyErr_Format(PyExc_TypeError,ex2err[EX2ERR_Invalid_Argument],"");

  rval = ex_get_var_param(exoid,"E", &num_elem_var);
  if ( rval < 0 )
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Inquire_Failed],"");

  rval = ex_inquire(exoid,EX_INQ_ELEM_BLK,&num_elem_blk,&fdmy,&cdmy);
  if ( rval < 0 )
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Inquire_Failed],"");

  sz = num_elem_blk*num_elem_var;

  tab = (PyArrayObject *)PyArray_EMPTY(1,&sz,PyArray_INT,PyArray_CORDER);
  if ( tab == NULL )
    return PyErr_Format(PyExc_MemoryError,ex2err[EX2ERR_Allocation_Error],"");

  ptr_tab = (int *)PyArray_DATA(tab);

  rval = ex_get_elem_var_tab(exoid,num_elem_blk,num_elem_var,ptr_tab);

  if ( rval < 0 ) {
    cleanup_po(1,tab);
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Library_Error],rval);
  }

  return PyArray_Return(tab);
}


//***************************************************************
// ex_put_elem_var_tab
//
static PyObject *ex2lib_ex_put_elem_var_tab(PyObject *self, PyObject *args) {
  int exoid, num_elem_blk, num_elem_var, rval, comp_ws, io_ws;
  PyObject *otab;
  PyArrayObject *tab;
  int *ptr_tab;

  if (!PyArg_ParseTuple(args,
                        "(iii)iiO:ex_put_elem_var_tab",
                        &exoid, &comp_ws, &io_ws, &num_elem_blk,
                        &num_elem_var, &otab)
      )
    return PyErr_Format(PyExc_TypeError,ex2err[EX2ERR_Invalid_Argument],"");

  tab = (PyArrayObject *)intarray(otab,1,1);
  if ( tab == NULL )
    return PyErr_Format(PyExc_MemoryError,ex2err[EX2ERR_Allocation_Error],"");

  ptr_tab = (int *)PyArray_DATA(tab);

  rval = ex_put_elem_var_tab(exoid,num_elem_blk,num_elem_var,ptr_tab);

  cleanup_po(1,tab);
  if ( rval < 0 )
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Library_Error],rval);

  return Py_BuildValue("i",rval);
}


//***************************************************************
// ex_get_elem_var
//
static PyObject *ex2lib_ex_get_elem_var(PyObject *self, PyObject *args) {
  int exoid, time_step, elem_var_index, elem_blk_id, num_elem_this_blk;
  int num_nodes_per_elem, num_attr, rval, comp_ws, io_ws, fdtype;
  npy_intp npy_num_elem_this_blk;
  char elem_type[MAX_STR_LENGTH+1];
  PyArrayObject *vals;
  void *ptr_vals;

  if (!PyArg_ParseTuple(args,
                        "(iii)iii:ex_get_elem_var",
                        &exoid, &comp_ws, &io_ws, &time_step,
                        &elem_var_index, &elem_blk_id)
      )
    return PyErr_Format(PyExc_TypeError,ex2err[EX2ERR_Invalid_Argument],"");

  rval = ex_get_elem_block(exoid,elem_blk_id,elem_type,&num_elem_this_blk,
                           &num_nodes_per_elem,&num_attr);
  if ( rval < 0 )
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Library_Error],rval);

  if ( comp_ws == 4 )
    fdtype = PyArray_FLOAT32;
  else
    fdtype = PyArray_FLOAT64;

  npy_num_elem_this_blk = num_elem_this_blk;

  vals = (PyArrayObject *)PyArray_EMPTY(1,&npy_num_elem_this_blk,fdtype,
                                        PyArray_CORDER);
  if ( vals == NULL )
    return PyErr_Format(PyExc_MemoryError,ex2err[EX2ERR_Allocation_Error],"");

  ptr_vals = (void *)PyArray_DATA(vals);

  rval = ex_get_elem_var(exoid,time_step,elem_var_index,elem_blk_id,
                         num_elem_this_blk,ptr_vals);

  if ( rval < 0 ) {
    cleanup_po(1,vals);
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Library_Error],rval);
  }

  return PyArray_Return(vals);
}


//***************************************************************
// ex_put_elem_var
//
static PyObject *ex2lib_ex_put_elem_var(PyObject *self, PyObject *args) {
  int exoid, time_step, elem_var_index, elem_blk_id, num_elem_this_blk;
  int rval, comp_ws, io_ws, fdtype;
  PyObject *ovals;
  PyArrayObject *vals;
  void *ptr_vals;

  if (!PyArg_ParseTuple(args,
                        "(iii)iiiiO:ex_put_elem_var",
                        &exoid, &comp_ws, &io_ws, &time_step,
                        &elem_var_index, &elem_blk_id,
                        &num_elem_this_blk, &ovals)
      )
    return PyErr_Format(PyExc_TypeError,ex2err[EX2ERR_Invalid_Argument],"");

  if ( comp_ws == 4 )
    fdtype = PyArray_FLOAT32;
  else
    fdtype = PyArray_FLOAT64;

  vals = (PyArrayObject *)fparray(ovals,fdtype,1,1);
  if ( vals == NULL )
    return PyErr_Format(PyExc_MemoryError,ex2err[EX2ERR_Allocation_Error],"");

  ptr_vals = (void *)PyArray_DATA(vals);

  rval = ex_put_elem_var(exoid,time_step,elem_var_index,elem_blk_id,
                         num_elem_this_blk,ptr_vals);

  cleanup_po(1,vals);
  if ( rval < 0 )
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Library_Error],rval);

  return Py_BuildValue("i",rval);
}


//***************************************************************
// ex_get_glob_vars
//
static PyObject *ex2lib_ex_get_glob_vars(PyObject *self, PyObject *args) {
  int exoid, time_step, num_glob_vars, rval, comp_ws, io_ws, fdtype;
  npy_intp npy_num_glob_vars;
  PyArrayObject *vals;
  void *ptr_vals;

  if (!PyArg_ParseTuple(args,
                        "(iii)i:ex_get_glob_vars",
                        &exoid, &comp_ws, &io_ws, &time_step)
      )
    return PyErr_Format(PyExc_TypeError,ex2err[EX2ERR_Invalid_Argument],"");

  rval = ex_get_var_param(exoid,"G",&num_glob_vars);
  if ( rval < 0 )
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Inquire_Failed],"");

  if ( comp_ws == 4 )
    fdtype = PyArray_FLOAT32;
  else
    fdtype = PyArray_FLOAT64;

  npy_num_glob_vars = num_glob_vars;

  vals = (PyArrayObject *)PyArray_EMPTY(1,&npy_num_glob_vars,fdtype,PyArray_CORDER);
  if ( vals == NULL )
    return PyErr_Format(PyExc_MemoryError,ex2err[EX2ERR_Allocation_Error],"");

  ptr_vals = (void *)PyArray_DATA(vals);

  rval = ex_get_glob_vars(exoid,time_step,num_glob_vars,ptr_vals);
  if ( rval < 0 ) {
    cleanup_po(1,vals);
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Library_Error],rval);
  }

  return PyArray_Return(vals);
}


//***************************************************************
// ex_put_glob_vars
//
static PyObject *ex2lib_ex_put_glob_vars(PyObject *self, PyObject *args) {
  int exoid, time_step, num_glob_vars, rval, comp_ws, io_ws, fdtype;
  PyObject *ovals;
  PyArrayObject *vals;
  void *ptr_vals;

  if (!PyArg_ParseTuple(args,
                        "(iii)iiO:ex_put_glob_vars",
                        &exoid, &comp_ws, &io_ws, &time_step, &num_glob_vars,
                        &ovals)
      )
    return PyErr_Format(PyExc_TypeError,ex2err[EX2ERR_Invalid_Argument],"");

  if ( comp_ws == 4 )
    fdtype = PyArray_FLOAT32;
  else
    fdtype = PyArray_FLOAT64;

  vals = (PyArrayObject *)fparray(ovals,fdtype,1,1);
  if ( vals == NULL )
    return PyErr_Format(PyExc_MemoryError,ex2err[EX2ERR_Allocation_Error],"");

  ptr_vals = (void *)PyArray_DATA(vals);

  rval = ex_put_glob_vars(exoid,time_step,num_glob_vars,ptr_vals);

  cleanup_po(1,vals);
  if ( rval < 0 )
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Library_Error],rval);

  return Py_BuildValue("i",rval);
}


//***************************************************************
// ex_get_nodal_var
//
static PyObject *ex2lib_ex_get_nodal_var(PyObject *self, PyObject *args) {
  int exoid, time_step, nodal_var_index, num_nodes, rval, comp_ws, io_ws;
  npy_intp npy_num_nodes;
  int fdtype;
  float fdmy;
  char cdmy;
  PyArrayObject *vals;
  void *ptr_vals;

  if (!PyArg_ParseTuple(args,
                        "(iii)ii:ex_get_nodal_var",
                        &exoid, &comp_ws, &io_ws, &time_step,
                        &nodal_var_index)
      )
    return PyErr_Format(PyExc_TypeError,ex2err[EX2ERR_Invalid_Argument],"");

  rval = ex_inquire(exoid,EX_INQ_NODES,&num_nodes,&fdmy,&cdmy);
  if ( rval < 0 )
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Inquire_Failed],"");

  if ( comp_ws == 4 )
    fdtype = PyArray_FLOAT32;
  else
    fdtype = PyArray_FLOAT64;

  npy_num_nodes = num_nodes;

  vals = (PyArrayObject *)PyArray_EMPTY(1,&npy_num_nodes,fdtype,PyArray_CORDER);
  if ( vals == NULL )
    return PyErr_Format(PyExc_MemoryError,ex2err[EX2ERR_Allocation_Error],"");

  ptr_vals = (void *)PyArray_DATA(vals);

  rval = ex_get_nodal_var(exoid,time_step,nodal_var_index,num_nodes,ptr_vals);
  if ( rval < 0 ) {
    cleanup_po(1,vals);
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Library_Error],rval);
  }

  return PyArray_Return(vals);
}


//***************************************************************
// ex_put_nodal_var
//
static PyObject *ex2lib_ex_put_nodal_var(PyObject *self, PyObject *args) {
  int exoid, time_step, nodal_var_index, num_nodes, rval, comp_ws, io_ws;
  int fdtype;
  PyObject *ovals;
  PyArrayObject *vals;
  void *ptr_vals;

  if (!PyArg_ParseTuple(args,
                        "(iii)iiiO:ex_put_nodal_var",
                        &exoid, &comp_ws, &io_ws, &time_step,
                        &nodal_var_index, &num_nodes, &ovals)
      )
    return PyErr_Format(PyExc_TypeError,ex2err[EX2ERR_Invalid_Argument],"");

  if ( comp_ws == 4 )
    fdtype = PyArray_FLOAT32;
  else
    fdtype = PyArray_FLOAT64;

  vals = (PyArrayObject *)fparray(ovals,fdtype,1,1);
  if ( vals == NULL )
    return PyErr_Format(PyExc_MemoryError,ex2err[EX2ERR_Allocation_Error],"");

  ptr_vals = (void *)PyArray_DATA(vals);

  rval = ex_put_nodal_var(exoid,time_step,nodal_var_index,num_nodes,ptr_vals);

  cleanup_po(1,vals);
  if ( rval < 0 )
    return PyErr_Format(PyExc_ValueError,ex2err[EX2ERR_Library_Error],rval);

  return Py_BuildValue("i",rval);
}


//***************************************************************
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


//***************************************************************
// WORKER FUNCTION fparray
// Wrapper around PyArray_ContiguousFromAny() that also forces
// a downcast from double to single precision if necessary.
//
static PyObject *fparray(PyObject *ob, int typen, int mnd, int mxd) {
  return PyArray_FromAny(ob,
                         PyArray_DescrFromType(typen),
                         mnd,
                         mxd,
                         NPY_DEFAULT | NPY_FORCECAST,
                         NULL);

}

//***************************************************************
// WORKER FUNCTION intarray
// Wrapper around PyArray_ContiguousFromAny() that also forces
// a downcast from long to 32 bit integer if necessary.
//
static PyObject *intarray(PyObject *ob, int mnd, int mxd) {
  return PyArray_FromAny(ob,
                         PyArray_DescrFromType(PyArray_INT),
                         mnd,
                         mxd,
                         NPY_DEFAULT | NPY_FORCECAST,
                         NULL);

}


static PyMethodDef ex2libmethods[] = {
  {"ex_create", ex2lib_ex_create, METH_VARARGS,
   "ex_create(path,cmode,comp_ws,io_ws).  Creates an ExodusII file."},
  {"ex_open", ex2lib_ex_open, METH_VARARGS, "Opens an ExodusII file."},
  {"ex_close", ex2lib_ex_close, METH_VARARGS, "Closes an ExodusII file."},
  {"ex_update", ex2lib_ex_update, METH_VARARGS, "Flushes an ExodusII file."},
  {"ex_get_init", ex2lib_ex_get_init, METH_VARARGS,
   "Read initialization parameter."},
  {"ex_put_init",ex2lib_ex_put_init, METH_VARARGS,
   "Write initialization parameters."},
  {"ex_get_qa", ex2lib_ex_get_qa, METH_VARARGS, "Reads QA records."},
  {"ex_put_qa", ex2lib_ex_put_qa, METH_VARARGS, "Write QA records."},
  {"ex_get_info", ex2lib_ex_get_info, METH_VARARGS,
   "Read information records."},
  {"ex_put_info", ex2lib_ex_put_info, METH_VARARGS,
   "Write information records."},
  {"ex_inquire", ex2lib_ex_inquire, METH_VARARGS,
   "Inquire ExodusII parameters."},
  {"ex_get_coord", ex2lib_ex_get_coord, METH_VARARGS,
   "Read nodal coordinates."},
  {"ex_put_coord", ex2lib_ex_put_coord, METH_VARARGS,
   "Write nodal coordinates."},
  {"ex_get_coord_names", ex2lib_ex_get_coord_names, METH_VARARGS,
   "Read coordinate names."},
  {"ex_put_coord_names", ex2lib_ex_put_coord_names, METH_VARARGS,
   "Write coordinate names."},
  {"ex_get_node_num_map", ex2lib_ex_get_node_num_map, METH_VARARGS,
   "Read node number map."},
  {"ex_put_node_num_map", ex2lib_ex_put_node_num_map, METH_VARARGS,
   "Write node number map."},
  {"ex_get_elem_num_map", ex2lib_ex_get_elem_num_map, METH_VARARGS,
   "Read element number map."},
  {"ex_put_elem_num_map", ex2lib_ex_put_elem_num_map, METH_VARARGS,
   "Write element number map."},
  {"ex_get_map", ex2lib_ex_get_map, METH_VARARGS, "Read element order map."},
  {"ex_put_map", ex2lib_ex_put_map, METH_VARARGS, "Write element order map."},
  {"ex_get_elem_block", ex2lib_ex_get_elem_block, METH_VARARGS,
   "Read element block parameters."},
  {"ex_put_elem_block", ex2lib_ex_put_elem_block, METH_VARARGS,
   "Write element block parameters."},
  {"ex_get_elem_blk_ids", ex2lib_ex_get_elem_blk_ids, METH_VARARGS,
   "Read element block ID's."},
  {"ex_get_elem_conn", ex2lib_ex_get_elem_conn, METH_VARARGS,
   "Read element block connectivity."},
  {"ex_put_elem_conn", ex2lib_ex_put_elem_conn, METH_VARARGS,
   "Write element block connectivity."},
  {"ex_get_elem_attr", ex2lib_ex_get_elem_attr, METH_VARARGS,
   "Read element block attributes."},
  {"ex_put_elem_attr", ex2lib_ex_put_elem_attr, METH_VARARGS,
   "Write element block attributes."},
  {"ex_get_node_set_param", ex2lib_ex_get_node_set_param, METH_VARARGS,
   "Read node set parameters."},
  {"ex_put_node_set_param", ex2lib_ex_put_node_set_param, METH_VARARGS,
   "Write node set parameters."},
  {"ex_get_node_set", ex2lib_ex_get_node_set, METH_VARARGS, "Read node set."},
  {"ex_put_node_set", ex2lib_ex_put_node_set, METH_VARARGS, "Write node set."},
  {"ex_get_node_set_dist_fact", ex2lib_ex_get_node_set_dist_fact, METH_VARARGS,
   "Read node set distribution factors."},
  {"ex_put_node_set_dist_fact", ex2lib_ex_put_node_set_dist_fact, METH_VARARGS,
   "Write node set distribution factors."},
  {"ex_get_node_set_ids", ex2lib_ex_get_node_set_ids, METH_VARARGS,
   "Read node set ID's."},
  {"ex_get_concat_node_sets", ex2lib_ex_get_concat_node_sets, METH_VARARGS,
   "Read concatenated node sets."},
  {"ex_put_concat_node_sets", ex2lib_ex_put_concat_node_sets, METH_VARARGS,
   "Write concatenated node sets."},
  {"ex_get_side_set_param", ex2lib_ex_get_side_set_param, METH_VARARGS,
   "Read side set parameters."},
  {"ex_put_side_set_param", ex2lib_ex_put_side_set_param, METH_VARARGS,
   "Write side set parameters."},
  {"ex_get_side_set", ex2lib_ex_get_side_set, METH_VARARGS,
   "Read side set"},
  {"ex_put_side_set", ex2lib_ex_put_side_set, METH_VARARGS,
   "Write side set"},
  {"ex_get_side_set_dist_fact", ex2lib_ex_get_side_set_dist_fact, METH_VARARGS,
   "Read side set distribution factors."},
  {"ex_put_side_set_dist_fact", ex2lib_ex_put_side_set_dist_fact, METH_VARARGS,
   "Write side set distribution factors."},
  {"ex_get_concat_side_sets", ex2lib_ex_get_concat_side_sets, METH_VARARGS,
   "Read concatenated side sets."},
  {"ex_put_concat_side_sets", ex2lib_ex_put_concat_side_sets, METH_VARARGS,
   "Write concatenated side sets."},
  {"ex_get_prop_names", ex2lib_ex_get_prop_names, METH_VARARGS,
   "Read property array names."},
  {"ex_put_prop_names", ex2lib_ex_put_prop_names, METH_VARARGS,
   "Write property array names."},
  {"ex_get_prop", ex2lib_ex_get_prop, METH_VARARGS,
   "Read object property."},
  {"ex_put_prop", ex2lib_ex_put_prop, METH_VARARGS,
   "Write object property."},
  {"ex_get_prop_array", ex2lib_ex_get_prop_array, METH_VARARGS,
   "Read object property array."},
  {"ex_put_prop_array", ex2lib_ex_put_prop_array, METH_VARARGS,
   "Write object property array."},
  {"ex_get_var_param", ex2lib_ex_get_var_param, METH_VARARGS,
   "Read results variables parameters."},
  {"ex_put_var_param", ex2lib_ex_put_var_param, METH_VARARGS,
   "Write results variables parameters."},
  {"ex_get_var_names", ex2lib_ex_get_var_names, METH_VARARGS,
   "Read results variables names."},
  {"ex_put_var_names", ex2lib_ex_put_var_names, METH_VARARGS,
   "Write results variables names."},
  {"ex_get_time", ex2lib_ex_get_time, METH_VARARGS,
   "Read time value for a time step."},
  {"ex_put_time", ex2lib_ex_put_time, METH_VARARGS,
   "Write time value for a time step."},
  {"ex_get_elem_var_tab", ex2lib_ex_get_elem_var_tab, METH_VARARGS,
   "Read element truth table."},
  {"ex_put_elem_var_tab", ex2lib_ex_put_elem_var_tab, METH_VARARGS,
   "Write element truth table."},
  {"ex_get_elem_var", ex2lib_ex_get_elem_var, METH_VARARGS,
   "Read element variable values at a time step."},
  {"ex_put_elem_var", ex2lib_ex_put_elem_var, METH_VARARGS,
   "Write element variable values at a time step."},
  {"ex_get_glob_vars", ex2lib_ex_get_glob_vars, METH_VARARGS,
   "Read global variables values at a time step."},
  {"ex_put_glob_vars", ex2lib_ex_put_glob_vars, METH_VARARGS,
   "Write global variables values at a time step."},
  {"ex_get_nodal_var", ex2lib_ex_get_nodal_var, METH_VARARGS,
   "Read nodal variable values at a time step."},
  {"ex_put_nodal_var", ex2lib_ex_put_nodal_var, METH_VARARGS,
   "Write nodal variable values at a time step."},
  {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initex2lib(void) {
  (void) Py_InitModule("ex2lib",ex2libmethods);
  import_array();
}


/*
 * Copyright (c) 2006 Sandia Corporation. Under the terms of Contract
 * DE-AC04-94AL85000 with Sandia Corporation, the U.S. Governement
 * retains certain rights in this software.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 * 
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 * 
 *     * Redistributions in binary form must reproduce the above
 *       copyright notice, this list of conditions and the following
 *       disclaimer in the documentation and/or other materials provided
 *       with the distribution.  
 * 
 *     * Neither the name of Sandia Corporation nor the names of its
 *       contributors may be used to endorse or promote products derived
 *       from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * 
 */
/*****************************************************************************
*
* exgini - ex_get_init
*
* entry conditions - 
*   input parameters:
*       int     exoid                   exodus file id
*
* exit conditions - 
*       char*   title                   title of file
*       int*    num_dim                 number of dimensions (per node)
*       int*    num_nodes               number of nodes
*       int*    num_elem                number of elements
*       int*    num_elem_blk            number of element blocks
*       int*    num_node_sets           number of node sets
*       int*    num_side_sets           numver of side sets
*
* revision history - 
*          David Thompson  - Moved to exginix.c (exgini.c now a special case)
*
*  $Id: exgini.c,v 1.5 2007/10/08 15:01:40 gdsjaar Exp $
*
*****************************************************************************/

#include "exodusII.h"
#include "exodusII_int.h"
#include <string.h>

/*
 * reads the initialization parameters from an opened EXODUS II file
 */

int ex_get_init (int   exoid,
                 char *title,
                 int  *num_dim,
                 int  *num_nodes,
                 int  *num_elem, 
                 int  *num_elem_blk,
                 int  *num_node_sets,
                 int  *num_side_sets)
{
  ex_init_params info;
  int errval;

  info.title[0] = '\0';
  errval = ex_get_init_ext( exoid, &info );
  if ( errval < 0 )
    {
      return errval;
    }

  *num_dim       = info.num_dim;
  *num_nodes     = info.num_nodes;
  *num_elem      = info.num_elem;
  *num_elem_blk  = info.num_elem_blk;
  *num_node_sets = info.num_node_sets;
  *num_side_sets = info.num_side_sets;
  strcpy( title, info.title );

  return (EX_NOERR);
}

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
* expoea - ex_put_one_elem_attr
*
* entry conditions - 
*   input parameters:
*       int     exoid                   exodus file id
*       int     elem_blk_id             element block id
*       int     attrib_index            index of attribute to write
*       float*  attrib                  array of attributes
*
* exit conditions - 
*
* revision history - 
*   20061003 - David Thompson - moved to ex_put_one_attr
*
*  $Id: expoea.c,v 1.4 2007/10/08 15:01:47 gdsjaar Exp $
*
*****************************************************************************/

#include "exodusII.h"
#include "exodusII_int.h"

/*!
 * writes the specified attribute for an element block
 */

int ex_put_one_elem_attr (int   exoid,
			  int   elem_blk_id,
			  int   attrib_index,
			  const void *attrib)
{
  return ex_put_one_attr( exoid, EX_ELEM_BLOCK, elem_blk_id, attrib_index, attrib );
}

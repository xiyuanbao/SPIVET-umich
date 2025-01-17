/*
 * Copyright (c) 2005 Sandia Corporation. Under the terms of Contract
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
* exgnam - ex_get_name
*
* entry conditions - 
*   input parameters:
*       int     exoid          exodus file id
*       const char *type       entity type - M, E, S   
*       int     entity_id      id of entity name to read 
*
* exit conditions - 
*       char*   name           ptr to name
*
* revision history - 
*
*  $Id: exgnam.c,v 1.4 2007/10/08 15:01:41 gdsjaar Exp $
*
*****************************************************************************/

#include "exodusII.h"
#include "exodusII_int.h"

/*
 * reads the specified entity name from the database
 */

int ex_get_name (int   exoid,
		 int   obj_type,
		 int   entity_id, 
		 char *name)
{
  int j, varid, ent_ndx;
  long num_entity, start[2];
  char *ptr;
  char errmsg[MAX_ERR_LENGTH];
  const char *routine = "ex_get_name";
   
  exerrval = 0;

  /* inquire previously defined dimensions and variables  */

  if (obj_type == EX_ELEM_BLOCK) {
    ex_get_dimension(exoid, DIM_NUM_EL_BLK, "element block", &num_entity, routine);
    varid = ncvarid (exoid, VAR_NAME_EL_BLK);
    ent_ndx = ex_id_lkup(exoid, VAR_ID_EL_BLK, entity_id);
  }
  else if (obj_type == EX_EDGE_BLOCK) {
    ex_get_dimension(exoid, DIM_NUM_ED_BLK, "edge block", &num_entity, routine);
    varid = ncvarid (exoid, VAR_NAME_ED_BLK);
    ent_ndx = ex_id_lkup(exoid, VAR_ID_ED_BLK, entity_id);
  }
  else if (obj_type == EX_FACE_BLOCK) {
    ex_get_dimension(exoid, DIM_NUM_FA_BLK, "face block", &num_entity, routine);
    varid = ncvarid (exoid, VAR_NAME_FA_BLK);
    ent_ndx = ex_id_lkup(exoid, VAR_ID_FA_BLK, entity_id);
  }
  else if (obj_type == EX_NODE_SET) {
    ex_get_dimension(exoid, DIM_NUM_NS, "nodeset", &num_entity, routine);
    varid = ncvarid (exoid, VAR_NAME_NS);
    ent_ndx = ex_id_lkup(exoid, VAR_NS_IDS, entity_id);
  }
  else if (obj_type == EX_SIDE_SET) {
    ex_get_dimension(exoid, DIM_NUM_SS, "sideset", &num_entity, routine);
    varid = ncvarid (exoid, VAR_NAME_SS);
    ent_ndx = ex_id_lkup(exoid, VAR_SS_IDS, entity_id);
  }
  else if (obj_type == EX_NODE_MAP) {
    ex_get_dimension(exoid, DIM_NUM_NM, "node map", &num_entity, routine);
    varid = ncvarid (exoid, VAR_NAME_NM);
    ent_ndx = ex_id_lkup(exoid, VAR_NM_PROP(1), entity_id);
  }
  else if (obj_type == EX_ELEM_MAP) {
    ex_get_dimension(exoid, DIM_NUM_EM, "element map", &num_entity, routine);
    varid = ncvarid (exoid, VAR_NAME_EM);
    ent_ndx = ex_id_lkup(exoid, VAR_EM_PROP(1), entity_id);
  }
  else {/* invalid variable type */
    exerrval = EX_BADPARAM;
    sprintf(errmsg, "Error: Invalid type specified in file id %d", exoid);
    ex_err(routine,errmsg,exerrval);
    return(EX_FATAL);
  }
   
  if (varid != -1) {
    /* If this is a null entity, then 'ent_ndx' will be negative.
     * We don't care in this routine, so make it positive and continue...
     */
    if (ent_ndx < 0) ent_ndx = -ent_ndx;
    
    /* read the name */
    start[0] = ent_ndx-1;
    start[1] = 0;
       
    j = 0;
    ptr = name;
       
    if (ncvarget1 (exoid, varid, start, ptr) == -1) {
      exerrval = ncerr;
      sprintf(errmsg,
	      "Error: failed to get entity name for id %d in file id %d",
	      ent_ndx, exoid);
      ex_err(routine,errmsg,exerrval);
      return (EX_FATAL);
    }
       
       
    while ((*ptr++ != '\0') && (j < MAX_STR_LENGTH)) {
      start[1] = ++j;
      if (ncvarget1 (exoid, varid, start, ptr) == -1) {
	exerrval = ncerr;
	sprintf(errmsg,
		"Error: failed to get name in file id %d", exoid);
	ex_err(routine,errmsg,exerrval);
	return (EX_FATAL);
      }
    }
    --ptr;
    if (ptr > name) {
      while (*(--ptr) == ' ');      /*    get rid of trailing blanks */
    }
    *(++ptr) = '\0';
  } else {
    /* Name variable does not exist on the database; probably since this is an
     * older version of the database.  Return an empty array...
     */
    name[0] = '\0';
  }
  return (EX_NOERR);
}

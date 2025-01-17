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
* expvan - ex_put_var_names
*
* entry conditions - 
*   input parameters:
*       int     exoid                   exodus file id
*       char*   var_type                variable type: G,N, or E
*       int     num_vars                # of variables to read
*       char*   var_names               ptr array of variable names
*
* exit conditions - 
*
* revision history - 
*
*  $Id: expvan.c,v 1.5 2007/10/08 15:01:48 gdsjaar Exp $
*
*****************************************************************************/

#include "exodusII.h"
#include "exodusII_int.h"
#include <string.h>
#include <ctype.h>

#define EX_PUT_NAMES(TNAME,DNUMVAR,VNAMES) \
     if ((ncdimid (exoid, DNUMVAR)) == -1) \
     { \
       if (ncerr == NC_EBADDIM) \
       { \
         exerrval = ncerr; \
         sprintf(errmsg, \
                "Error: no " TNAME " variables defined in file id %d", \
                 exoid); \
         ex_err("ex_put_var_names",errmsg,exerrval); \
       } \
       else \
       { \
         exerrval = ncerr; \
         sprintf(errmsg, \
             "Error: failed to locate number of " TNAME " variables in file id %d", \
                 exoid); \
         ex_err("ex_put_var_names",errmsg,exerrval); \
       } \
       return(EX_FATAL);  \
     } \
 \
     if ((varid = ncvarid (exoid, VNAMES)) == -1) \
     { \
       if (ncerr == NC_ENOTVAR) \
       { \
         exerrval = ncerr; \
         sprintf(errmsg, \
                "Error: no " TNAME " variable names defined in file id %d", \
                 exoid); \
         ex_err("ex_put_var_names",errmsg,exerrval); \
       } \
       else \
       { \
         exerrval = ncerr; \
         sprintf(errmsg, \
                "Error: " TNAME " name variable names not found in file id %d", \
                 exoid); \
         ex_err("ex_put_var_names",errmsg,exerrval); \
       } \
       return(EX_FATAL); \
     }

/*!
 * writes the names of the results variables to the database
 */

int ex_put_var_names (int   exoid,
                      const char *var_type,
                      int   num_vars,
                      char* var_names[])
{
   int i, varid; 
   long  start[2], count[2];
   char errmsg[MAX_ERR_LENGTH];
   int vartyp;

   exerrval = 0; /* clear error code */

   vartyp = tolower( *var_type );
   switch (vartyp) {
   case 'g':
     EX_PUT_NAMES(     "global",DIM_NUM_GLO_VAR,  VAR_NAME_GLO_VAR);
     break;
   case 'n':
     EX_PUT_NAMES(      "nodal",DIM_NUM_NOD_VAR,  VAR_NAME_NOD_VAR);
     break;
   case 'l':
     EX_PUT_NAMES(       "edge",DIM_NUM_EDG_VAR,  VAR_NAME_EDG_VAR);
     break;
   case 'f':
     EX_PUT_NAMES(       "face",DIM_NUM_FAC_VAR,  VAR_NAME_FAC_VAR);
     break;
   case 'e':
     EX_PUT_NAMES(    "element",DIM_NUM_ELE_VAR,  VAR_NAME_ELE_VAR);
     break;
   case 'm':
     EX_PUT_NAMES(   "node set",DIM_NUM_NSET_VAR, VAR_NAME_NSET_VAR);
     break;
   case 'd':
     EX_PUT_NAMES(   "edge set",DIM_NUM_ESET_VAR, VAR_NAME_ESET_VAR);
     break;
   case 'a':
     EX_PUT_NAMES(   "face set",DIM_NUM_FSET_VAR, VAR_NAME_FSET_VAR);
     break;
   case 's':
     EX_PUT_NAMES(   "side set",DIM_NUM_SSET_VAR, VAR_NAME_SSET_VAR);
     break;
   case 't':
     EX_PUT_NAMES("element set",DIM_NUM_ELSET_VAR,VAR_NAME_ELSET_VAR);
     break;
   default:
     exerrval = EX_BADPARAM;
     sprintf(errmsg,
            "Error: Invalid variable type %c specified in file id %d",
             *var_type, exoid);
     ex_err("ex_put_var_names",errmsg,exerrval);
     return(EX_FATAL);
   }



/* write EXODUS variable names */

   for (i=0; i<num_vars; i++)
   {
     start[0] = i;
     start[1] = 0;

     count[0] = 1;
     count[1] = strlen(var_names[i]) + 1;

     if (ncvarput (exoid, varid, start, count, (void*) var_names[i]) == -1)
     {
       exerrval = ncerr;
       sprintf(errmsg,
               "Error: failed to store variable names in file id %d",
                exoid);
       ex_err("ex_put_var_names",errmsg,exerrval);
       return (EX_FATAL);
     }
   }

   return(EX_NOERR);

}

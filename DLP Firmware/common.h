/***************************************************************************** 
**
**             TEXAS INSTRUMENTS PROPRIETARY INFORMATION
**
**  (c) Copyright, Texas Instruments Incorporated, 2008
**      All Rights Reserved.
**
**  Property of Texas Instruments Incorporated. Restricted Rights -
**  Use, duplication, or disclosure is subject to restrictions set
**  forth in TI's program license agreement and associated documentation.
******************************************************************************/
/*************************************************************
* THIS PROGRAM IS PROVIDED "AS IS." TI MAKES NO WARRANTIES OR
* REPRESENTATIONS, EITHER EXPRESS, IMPLIED OR STATUTORY,
* INCLUDING ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS
* FOR A PARTICULAR PURPOSE, LACK OF VIRUSES, ACCURACY OR
* COMPLETENESS OF RESPONSES, RESULTS AND LACK OF NEGLIGENCE.
* TI DISCLAIMS ANY WARRANTY OF TITLE, QUIET ENJOYMENT, QUIET
* POSSESSION, AND NON-INFRINGEMENT OF ANY THIRD PARTY
* INTELLECTUAL PROPERTY RIGHTS WITH REGARD TO THE PROGRAM OR
* YOUR USE OF THE PROGRAM.
*
* IN NO EVENT SHALL TI BE LIABLE FOR ANY SPECIAL, INCIDENTAL,
* CONSEQUENTIAL OR INDIRECT DAMAGES, HOWEVER CAUSED, ON ANY
* THEORY OF LIABILITY AND WHETHER OR NOT TI HAS BEEN ADVISED
* OF THE POSSIBILITY OF SUCH DAMAGES, ARISING IN ANY WAY OUT
* OF THIS AGREEMENT, THE PROGRAM, OR YOUR USE OF THE PROGRAM.
* EXCLUDED DAMAGES INCLUDE, BUT ARE NOT LIMITED TO, COST OF
* REMOVAL OR REINSTALLATION, COMPUTER TIME, LABOR COSTS, LOSS
* OF GOODWILL, LOSS OF PROFITS, LOSS OF SAVINGS, OR LOSS OF
* USE OR INTERRUPTION OF BUSINESS.  IN NO EVENT WILL TI'S
* AGGREGATE LIABILITY UNDER THIS AGREEMENT OR ARISING OUT OF
* YOUR USE OF THE PROGRAM EXCEED FIVE HUNDRED DOLLARS
* (U.S.$500).
*
* Unless otherwise stated, the Program is written and copyrighted
* by Texas Instruments is distributed as "freeware."  You may,
* only under TI's copyright in the Program, use and modify the
* Program without any charge or restriction.  You may
* distribute to third parties, provided that you transfer a
* copy of this license to the third party and the third party
* agrees to these terms by its first use of the Program.  In
* jurisdictions in which use is not deemed acceptance of these
* terms, no license is granted and no use is permitted.  You
* must reproduce the copyright notice and any other legend of
* ownership on each copy or partial copy of the Program.
*
* You acknowledge and agree that the Program contains
* copyrighted material, trade secrets and other TI proprietary
* information and is protected by copyright laws,
* international copyright treaties, and trade secret laws, as
* well as other intellectual property laws.  You agree that in
* no event will you alter, remove or destroy any copyright
* notice included in the Program.  TI reserves all rights not
* specifically granted under this license.  Except as
* specifically provided herein, nothing in this agreement
* shall be construed as conferring upon you, by implication,
* estoppel, or otherwise, any license or other right under any
* TI patents, copyrights or trade secrets.
*************************************************************/
#ifndef __COMMON_H
#define __COMMON_H

#ifdef __cplusplus
extern "C" {
#endif

/* typedefs for portability */
typedef unsigned       char uint08;
typedef   signed       char  int08;
typedef unsigned        int uint16;
typedef   signed        int  int16;
typedef unsigned       long uint32;
typedef   signed       long  int32;
typedef unsigned       char   BOOL;

#define TRUE    1
#define FALSE   0

#define ENABLE  1
#define DISABLE 0
//#define FAIL    0
#define PASS    1
#define NIL     0
#ifndef NULL
#  define NULL  0
#endif

/** Extract 8-bit value from 16-bit field */
#define LS_BYTE(x)  ((uint08)x)
#define MS_BYTE(x)  ((uint08)(x>>8))

/** Extract 16-bit value from 32-bit field */
#define LS_WORD(x)  ((uint16)x)
#define MS_WORD(x)  ((uint16)(x>>16))



#ifdef __cplusplus		/* matches __cplusplus construct above */
}
#endif

#endif /* __COMMON_H */

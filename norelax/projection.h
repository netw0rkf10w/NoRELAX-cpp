#ifndef PROJECTION_H
#define PROJECTION_H

/// Modify by D. Khue Le-Huu
/// The original file can be found on the author's website.
/// The original copyright is below.

/*
 #  File            : condat_simplexproj.c 
 #
 #  Version History : 1.0, Aug. 15, 2014 
 #
 #  Author          : Laurent Condat, PhD, CNRS research fellow in France.
 #
 #  Description     : This file contains an implementation in the C language
 #                    of algorithms described in the research paper:
 #	
 #                    L. Condat, "Fast Projection onto the Simplex and the
 #                    l1 Ball", preprint Hal-01056171, 2014.
 #
 #                    This implementation comes with no warranty: due to the
 #                    limited number of tests performed, there may remain
 #                    bugs. In case the functions would not do what they are
 #                    supposed to do, please email the author (contact info
 #                    to be found on the web).
 #
 #                    If you use this code or parts of it for any purpose,
 #                    the author asks you to cite the paper above or, in 
 #                    that event, its published version. Please email him if 
 #                    the proposed algorithms were useful for one of your 
 #                    projects, or for any comment or suggestion.
 #
 #  Usage rights    : Copyright Laurent Condat.
 #                    This file is distributed under the terms of the CeCILL
 #                    licence (compatible with the GNU GPL), which can be
 #                    found at the URL "http://www.cecill.info".
 #
 #  This software is governed by the CeCILL license under French law and
 #  abiding by the rules of distribution of free software. You can  use,
 #  modify and or redistribute the software under the terms of the CeCILL
 #  license as circulated by CEA, CNRS and INRIA at the following URL :
 #  "http://www.cecill.info".
 #
 #  As a counterpart to the access to the source code and rights to copy,
 #  modify and redistribute granted by the license, users are provided only
 #  with a limited warranty  and the software's author,  the holder of the
 #  economic rights,  and the successive licensors  have only  limited
 #  liability.
 #
 #  In this respect, the user's attention is drawn to the risks associated
 #  with loading,  using,  modifying and/or developing or reproducing the
 #  software by the user in light of its specific status of free software,
 #  that may mean  that it is complicated to manipulate,  and  that  also
 #  therefore means  that it is reserved for developers  and  experienced
 #  professionals having in-depth computer knowledge. Users are therefore
 #  encouraged to load and test the software's suitability as regards their
 #  requirements in conditions enabling the security of their systems and/or
 #  data to be ensured and,  more generally, to use and operate it in the
 #  same conditions as regards security.
 #
 #  The fact that you are presently reading this means that you have had
 #  knowledge of the CeCILL license and that you accept its terms.
*/


/* This code was compiled using
gcc -march=native -O2 condat_simplexproj.c -o main  -I/usr/local/include/ 
	-lm -lgsl  -L/usr/local/lib/
On my machine, gcc is actually a link to the compiler Apple LLVM version 5.1 
(clang-503.0.40) */


/* The following functions are implemented:
1) simplexproj_algo1
2) simplexproj_algo2
3) simplexproj_algo3_pivotrand
4) simplexproj_algo3_pivotmedian
5) simplexproj_Duchi
6) simplexproj_algo4
7) simplexproj_Condat (proposed algorithm)
All these functions take the same parameters. They project the vector y onto
the closest vector x of same length (parameter N in the paper) with x[n]>=0,
n=0..N-1, and sum_{n=0}^{N-1}x[n]=a. 
We can have x==y (projection done in place). If x!=y, the arrays x and y must
not overlap, as x is used for temporary calculations before y is accessed.
We must have length>=1 and a>0. 
*/

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>
//#include <gsl/gsl_rng.h>
//#include <gsl/gsl_randist.h>

#define datatype double /* type of the elements in y */


/* Algorithm using partitioning with respect to a pivot, chosen randomly, 
as given by Duchi et al. in "Efficient Projections onto the l1-Ball for 
Learning in High Dimensions" */
//static void simplexproj_Duchi(const datatype* y, datatype* x,
//const int length, const double a) {
//    datatype*	auxlower = (x==y ? (datatype*)malloc(length*sizeof(datatype)) : x);
//    datatype*	auxupper = (datatype*)malloc(length*sizeof(datatype));
//    datatype*	aux=auxlower;
//    datatype 	pivot;
//    int 	auxlowerlength=0;
//    int		auxupperlength=1;
//    int		upperlength;
//    int		auxlength;
//    int		i=0;
//    int 	pospivot=(int)(rand() / (((double)RAND_MAX+1.0)/length));
//    double	tauupper;
//    double	tau=(pivot=y[pospivot])-a;
//    while (i<pospivot)
//        if (y[i]<pivot)
//            auxlower[auxlowerlength++]=y[i++];
//        else {
//            auxupper[auxupperlength++]=y[i];
//            tau+=(y[i++]-tau)/auxupperlength;
//        }
//    i++;
//    while (i<length)
//        if (y[i]<pivot)
//            auxlower[auxlowerlength++]=y[i++];
//        else {
//            auxupper[auxupperlength++]=y[i];
//            tau+=(y[i++]-tau)/auxupperlength;
//        }
//    if (tau<pivot) {
//        upperlength=auxupperlength;
//        tauupper=tau;
//        auxlength=auxlowerlength;
//    } else {
//        tauupper=0.0;
//        upperlength=0;
//        aux=auxupper+1;
//        auxlength=auxupperlength-1;
//    }
//    while (auxlength>0) {
//        pospivot=(int)(rand() / (((double)RAND_MAX+1.0)/auxlength));
//        if (upperlength==0)
//            tau=(pivot=aux[pospivot])-a;
//        else
//            tau=tauupper+((pivot=aux[pospivot])-tauupper)/(upperlength+1);
//        i=0;
//        auxlowerlength=0;
//        auxupperlength=1;
//        while (i<pospivot)
//            if (aux[i]<pivot)
//                auxlower[auxlowerlength++]=aux[i++];
//            else {
//                auxupper[auxupperlength++]=aux[i];
//                tau+=(aux[i++]-tau)/(upperlength+auxupperlength);
//            }
//        i++;
//        while (i<auxlength)
//            if (aux[i]<pivot)
//                auxlower[auxlowerlength++]=aux[i++];
//            else {
//                auxupper[auxupperlength++]=aux[i];
//                tau+=(aux[i++]-tau)/(upperlength+auxupperlength);
//            }
//        if (tau<pivot) {
//            upperlength+=auxupperlength;
//            tauupper=tau;
//            auxlength=auxlowerlength;
//            aux=auxlower;
//        } else {
//            aux=auxupper+1;
//            auxlength=auxupperlength-1;
//        }
//    }
//    for (i=0; i<length; i++)
//        x[i]=(y[i]>tau ? y[i]-tauupper : 0.0);
//    if (x==y) free(auxlower);
//    free(auxupper);
//}


/* Proposed algorithm */
static void simplexproj_Condat(const datatype* y, datatype* x,
const int length, const double a) {
    datatype*	aux = (x==y ? (datatype*)malloc(length*sizeof(datatype)) : x);
    datatype*  aux0=aux;
    int		auxlength=1;
    int		auxlengthold=-1;
    double	tau=(*aux=*y)-a;
    int 	i=1;
    for (; i<length; i++)
        if (y[i]>tau) {
            if ((tau+=((aux[auxlength]=y[i])-tau)/(auxlength-auxlengthold))
            <=y[i]-a) {
                tau=y[i]-a;
                auxlengthold=auxlength-1;
            }
            auxlength++;
        }
    if (auxlengthold>=0) {
        auxlength-=++auxlengthold;
        aux+=auxlengthold;
        while (--auxlengthold>=0)
            if (aux0[auxlengthold]>tau)
                tau+=((*(--aux)=aux0[auxlengthold])-tau)/(++auxlength);
    }
    do {
        auxlengthold=auxlength-1;
        for (i=auxlength=0; i<=auxlengthold; i++)
            if (aux[i]>tau)
                aux[auxlength++]=aux[i];
            else
                tau+=(tau-aux[i])/(auxlengthold-i+auxlength);
    } while (auxlength<=auxlengthold);
    for (i=0; i<length; i++)
        x[i]=(y[i]>tau ? y[i]-tau : 0.0);
    if (x==y) free(aux0);
}




#endif // PROJECTION_H

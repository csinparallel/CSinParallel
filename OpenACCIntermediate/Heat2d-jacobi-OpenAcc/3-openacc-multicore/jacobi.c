/* Copyright (c) 2012, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "laplace2d.h"  
#include "../utils/getCommandLine.h"

int main(int argc, char** argv)
{
    // default size of our 'mesh'
    int n = 4096;
    int m = 4096;
    int iter_max = 1000;
    int verbose = 0;      // not used yet
    int numThreads = 1;

    getArguments(argc, argv, &n, &m, &iter_max, &verbose, &numThreads);

    // printf("numThreads = %d\n", numThreads);
    
    // acc_set_num_cores(numThreads);   // not available; we'll use default threads
    
    const double tol = 1.0e-6;  // small enough that even large iterations will run
    double error = 1.0;

    double *restrict A    = (double*)malloc(sizeof(double)*n*m);
    double *restrict Anew = (double*)malloc(sizeof(double)*n*m);
    // restrict tells the compiler that our code has no other pointers that
    // point to these arrays. It is used for optimization by the compiler.
    
    initialize(A, Anew, m, n);  
        
    printf("Jacobi relaxation Calculation: %d x %d mesh\n", n, m);
    
    double st = omp_get_wtime();
    int iter = 0;
   
    while ( error > tol && iter < iter_max )
    {
        error = calcNext(A, Anew, m, n);
        swap(A, Anew, m, n);

        if(iter % 100 == 0) printf("%5d, %0.6f\n", iter, error);
        
        iter++;

    }

    double runtime = omp_get_wtime() - st;
 
    printf("for %d iterations, total time: %f s\n", iter, runtime);

    deallocate(A, Anew);

    return 0;
}

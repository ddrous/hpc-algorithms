#include "utils.hpp"
#include "timer.hpp"
#include <iostream>
#include <random>
#include <vector>
#include <cassert>
#include <omp.h>
#include <memory>
#include <cstring>
#include <array>

// #include "cblas.h"
#include <cblas.h>

//#define SOLUTION
typedef double decimal;
// typedef float decimal;


int main(int argc, char** argv){
    if(argc != 2 || (argc == 2 && (strcmp(argv[1], "-check") !=0 && strcmp(argv[1], "-no-check") && strcmp(argv[1], "-no-optim")))){
        std::cout << "Should be\n";
        std::cout << argv[0] << " -check #to check the result\n";
        std::cout << argv[0] << " -no-check #to avoid checking the result\n";   
        std::cout << argv[0] << " -no-optim #to avoid running the optimized code\n";  
        return 1;     
    }

    const bool checkRes = (strcmp(argv[1], "-no-check") != 0);
    const bool runOptim = (strcmp(argv[1], "-no-optim") != 0);

    const long int N = 1024;        // 1024
    decimal* A = (decimal*)aligned_alloc(64, N * N * sizeof(decimal));
    memset(A, 0, N * N * sizeof(decimal));
    decimal* B = (decimal*)aligned_alloc(64, N * N * sizeof(decimal));
    memset(B, 0, N * N * sizeof(decimal));
    decimal* C = (decimal*)aligned_alloc(64, N * N * sizeof(decimal));
    memset(C, 0, N * N * sizeof(decimal));
    decimal* COptim = (decimal*)aligned_alloc(64, N * N * sizeof(decimal));
    memset(COptim, 0, N * N * sizeof(decimal));
   
    {        
        std::mt19937 gen(0);
        std::uniform_real_distribution<decimal> dis(0, 1);
        
        for(long int i = 0 ; i < N ; ++i){
            for(long int j = 0 ; j < N ; ++j){
                A[i*N+j] = dis(gen);
                B[j*N+i] = dis(gen);
            }
        }
    }   
    
    Timer timerNoOptim;
    if(checkRes){
        for(long int k = 0 ; k < N ; ++k){
            for(long int j = 0 ; j < N ; ++j){
                for(long int i = 0 ; i < N ; ++i){
                    C[i*N+j] += A[i*N+k] * B[j*N+k];
                }
            }
        }
    }
    timerNoOptim.stop();
    
    Timer timerWithOptim;
    if(runOptim){
        /******************** 8.1 Improve the memory access *******************/
        // 8.1 Inversing the loops
        // for(long int i = 0 ; i < N ; ++i){
        //     for(long int j = 0 ; j < N ; ++j){
        //         for(long int k = 0 ; k < N ; ++k){
        //             COptim[i*N+j] += A[i*N+k] * B[j*N+k];
        //         }
        //     }
        // }
        /***********************************************************************/

        /********************** 8.2 Parallelize the code ************************/
        // omp_set_num_threads(64);
        // std::cout << "num threads  " << omp_get_max_threads() << std::endl;
        // #pragma omp parallel for collapse(2)
        // for(long int i = 0 ; i < N ; ++i){
        //     for(long int j = 0 ; j < N ; ++j){
        //         for(long int k = 0 ; k < N ; ++k){
        //             COptim[i*N+j] += A[i*N+k] * B[j*N+k];
        //         }
        //     }
        // }
        /***********************************************************************/

        /******************** 8.3 Vectorize the inner loop *********************/
        // First version - Without Pointers 
        // #pragma omp parallel for collapse(1)
        // for(long int i = 0 ; i < N ; ++i){
        //     for(long int j = 0 ; j < N ; ++j){
        //         double sum = 0;
        //         #pragma omp simd reduction(+: sum) aligned(A, B: 64) safelen(N) simdlen(64)
        //         for(long int k = 0 ; k < N ; ++k){
        //              sum += A[i*N+k] * B[j*N+k];
        //         }
        //         COptim[i*N+j] = sum;
        //     }
        // }

        // // Second version - With Pointers
        // double* ptr1 = &A[0];
        // double* ptr2 = &B[0];
        // #pragma omp parallel for collapse(1)
        // for(long int i = 0 ; i < N ; ++i){
        //     for(long int j = 0 ; j < N ; ++j){
        //         double sum = 0;
        //         #pragma omp simd reduction(+: sum) aligned(ptr1, ptr2: 64) safelen(N) simdlen(64)
        //         for(long int k = 0 ; k < N ; ++k){
        //              sum += *(ptr1 + i*N+k) * *(ptr2 + j*N+k);
        //         }
        //         COptim[i*N+j] = sum;
        //     }
        // }
        /**********************************************************************/

        /************************ 8.4 a) float vs. double *********************/
        decimal* ptr1 = &A[0];
        decimal* ptr2 = &B[0];
        #pragma omp parallel for collapse(1)
        for(long int i = 0 ; i < N ; ++i){
            for(long int j = 0 ; j < N ; ++j){
                decimal sum = 0;
                #pragma omp simd reduction(+: sum) aligned(ptr1, ptr2: 64) safelen(N) simdlen(64)
                for(long int k = 0 ; k < N ; ++k){
                     sum += *(ptr1 + i*N+k) * *(ptr2 + j*N+k);
                }
                COptim[i*N+j] = sum;
            }
        }

        /************************** 8.4 b) BLAS implementationm ****************/
        // cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1.0, A, N, B, N, 0.0, COptim, N);


    }
    timerWithOptim.stop();
    
    if(checkRes){
        std::cout << ">> Without Optim : " << timerNoOptim.getElapsed() << std::endl;
        if(runOptim){
            for(long int i = 0 ; i < N ; ++i){
                for(long int j = 0 ; j < N ; ++j){
                    CheckEqual(C[i*N+j],COptim[i*N+j]);
                }
            }
        }
    }
    if(runOptim){
        std::cout << ">> With Optim : " << timerWithOptim.getElapsed() << std::endl;
    }
    
    free(A);
    free(B);
    free(C);
    free(COptim);
    
    return 0;
}
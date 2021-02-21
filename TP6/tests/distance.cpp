
#include <cuda.h>
#include <cmath>


/* Example kernel */
// __global__ void fill(int* blockMat, int* threadMat, const int N){
//     const int i_start = blockIdx.y*blockDim.y+threadIdx.y;      // we have to compute the global index
//     const int j_start = blockIdx.x*blockDim.x+threadIdx.x;

//     const int i_inc = gridDim.y*blockDim.y;
//     const int j_inc = gridDim.x*blockDim.x;

//     for(int i = i_start ; i < N ; i += i_inc){
//         for(int j = j_start ; j < N ; j += j_inc){
//             blockMat[i*N+j] = blockIdx.y*gridDim.x + blockIdx.x;
//             threadMat[i*N+j] = i_start*gridDim.x*blockDim.x + j_start;          // put a random number in the matrix
//         }
//     }
// }


#include <memory>
#include <iostream>
#include <iomanip>


/* CUDA grid mapping notes*/
/* gridDim.x (is the number of blocks), blockDim.x (is the number of threads in each block), blockIdx.x (is the index of the current block within the grid), and threadIdx.x (is the index of the current thread inside the block)  */

__global__ void fillDistance(int* blockMat, int* threadMat, const int N){

    /** Thread row and column global ids (// https://stackoverflow.com/questions/29858234/cuda-2d-array-mapping). 
     *  NOTE! These ids are defined only for the grid (2x2) and block(2x2) indicated in the host code.
     *  If the number of total number of threads is not enough to process one element in the array (NxN < 2*2 x 2*2),
     *  some tthreads will have to work on more than one element. 
     *  A loop will be needed to manually compute the furute_ids of the elments the current thread will work on .  
     *  Hence the need for and incrementation aong the rows, and along the columns.  
     */
    const int i_start = blockIdx.y*blockDim.y+threadIdx.y;      // global row id
    const int j_start = blockIdx.x*blockDim.x+threadIdx.x;      // global col id 

    const int i_inc = gridDim.y*blockDim.y;
    const int j_inc = gridDim.x*blockDim.x;

    double i_center = (N) / 2;
    double j_center = (N) / 2;

    // To computed ids of threads added by the CudaCPU function
    for(int i = i_start ; i < N ; i += i_inc){
        for(int j = j_start ; j < N ; j += j_inc){
            blockMat[i*N+j] = blockIdx.y*gridDim.x + blockIdx.x;

            /* Check that the ids are correct, especially when the CUDA kernel must create more threads than needed
               (the number of elements in the array is not evenly divisible by the block size) */
            if (i < N && j < N)
                threadMat[i*N+j] = (int)sqrt(pow(i-i_center, 2) + pow(j-j_center, 2));
        }
    }
}


int main(){
    const int N = 20;

    // Malloc and copy device arrays. These arrays are destined to be used by the Cuda kernel
    int* cuBlockMat;
    cudaMalloc(&cuBlockMat, sizeof(int)*N*N);           // Allocate the space on the GPU 
    assert(cudaGetLastError() == cudaSuccess);
    cudaMemset(cuBlockMat, ~0, sizeof(int)*N*N);        // Put 0 in the array. Since we cannot do cuBlockMat[0] (it's forbidden), we must use cudaMemset and cudaMemcpy
    assert(cudaGetLastError() == cudaSuccess);

    int* cuThreadMat;
    cudaMalloc(&cuThreadMat, sizeof(int)*N*N);
    assert(cudaGetLastError() == cudaSuccess);
    cudaMemset(cuThreadMat, ~0, sizeof(int)*N*N);
    assert(cudaGetLastError() == cudaSuccess);

    // TODO Create a cuDistance array (NO NEED!, all is done in the kernel and returned in cuThreadMat)
    // TODO call your kernel

    // This functions (NOT DEFINED BY CUDA), apparently a single thread to work on more than one element of the array
    CudaCpu(dim3(2,4), dim3(5,2), fillDistance, cuBlockMat, cuThreadMat, N);    // Run the fillDistance kernel and put the results in the device arrays. The 1st and 2nd arguments indicate the number of blocks per grid and threads per block repectively.
    
    // Get back results
    std::unique_ptr<int[]> blockMat(new int[N*N]);
    cudaMemcpy(blockMat.get(), cuBlockMat, sizeof(int)*N*N, cudaMemcpyDeviceToHost);    // copy from gpu to cpu
    assert(cudaGetLastError() == cudaSuccess);

    std::unique_ptr<int[]> threadMat(new int[N*N]);
    cudaMemcpy(threadMat.get(), cuThreadMat, sizeof(int)*N*N, cudaMemcpyDeviceToHost);
    assert(cudaGetLastError() == cudaSuccess);

    // TODO retreive the results (see cuThreadMat)
    // Free
    cudaFree(cuBlockMat);
    assert(cudaGetLastError() == cudaSuccess);
    cudaFree(cuThreadMat);
    assert(cudaGetLastError() == cudaSuccess);

    // TODO free your array (see cuThreadMat)

    // Print result
    // std::cout << "blockMat :" << std::endl;
    // for(int i = 0 ; i < N ; ++i){
    //     for(int j = 0 ; j < N ; ++j){
    //         std::cout << std::setw(3) << blockMat[i*N+j] << " ";
    //     }
    //     std::cout << "\n";
    // }

    std::cout << "threadMat :" << std::endl;
    for(int i = 0 ; i < N ; ++i){
        for(int j = 0 ; j < N ; ++j){
            std::cout << std::setw(3) << threadMat[i*N+j] << " ";
        }
        std::cout << "\n";
    }

    // TODO print the content of the array (see cuThreadMat)

    return 0;
}
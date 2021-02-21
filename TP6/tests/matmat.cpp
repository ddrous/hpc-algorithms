
#include <cuda.h>

__global__ void matmat(float* A, float* B, float* C, const int N){

    // TODO
    const int i_start = blockIdx.y*blockDim.y+threadIdx.y;      // global row id
    const int j_start = blockIdx.x*blockDim.x+threadIdx.x;      // global col id 

    const int i_inc = gridDim.y*blockDim.y;
    const int j_inc = gridDim.x*blockDim.x;

    // To compute the matmat product
    for(int i = i_start ; i < N ; i += i_inc){
        for(int j = j_start ; j < N ; j += j_inc){
            float tmp = 0;  // To minimize device memory access
            for(int k = 0 ; k < N ; k += 1){
                tmp += B[i*N+k]*C[k*N+j];
            }
            A[i*N+j] = tmp;
        }
    }
}


#include <memory>
#include <iostream>



/* The goal is to perform the matmat operation Z=BC */
int main(){
    const int N = 100;

    std::unique_ptr<float[]> A(new float[N*N]);
    std::unique_ptr<float[]> B(new float[N*N]);
    std::unique_ptr<float[]> C(new float[N*N]);

    for(int i = 0 ; i < N ; ++i){
        for(int j = 0 ; j < N ; ++j){
            B[i*N+j] = float(i * j);
            C[i*N+j] = float(i + j);
        }
    }

    for(int i = 0 ; i < N ; ++i){
        for(int j = 0 ; j < N ; ++j){
            for(int k = 0 ; k < N ; ++k){
               A[i*N+j] += B[i*N+k] * C[k*N+j];
            }
        }
    }

    // Will get results from the computation
    std::unique_ptr<float[]> A_from_CUDA(new float[N*N]());

    // TODO allocate cuA cuB and cuC
    float* cuA, *cuB, *cuC;

    cudaMalloc(&cuA, sizeof(float)*N*N); 
    assert(cudaGetLastError() == cudaSuccess);
    cudaMalloc(&cuB, sizeof(float)*N*N); 
    assert(cudaGetLastError() == cudaSuccess);
    cudaMalloc(&cuC, sizeof(float)*N*N); 
    assert(cudaGetLastError() == cudaSuccess);

    // TODO copy B and C to cuB and cuC
    cudaMemcpy(cuB, B.get(), N*N*sizeof(float), cudaMemcpyHostToDevice);
    assert(cudaGetLastError() == cudaSuccess);
    cudaMemcpy(cuC, C.get(), N*N*sizeof(float), cudaMemcpyHostToDevice);
    assert(cudaGetLastError() == cudaSuccess);

    // TODO init cuA to zero
    cudaMemset(cuA, ~0, sizeof(float)*N*N);
    assert(cudaGetLastError() == cudaSuccess);

    // TODO call your kernel
    CudaCpu(dim3(2,2), dim3(2,2), matmat, cuA, cuB, cuC, N);

    // TODO copy back your result from cuA into A_from_CUDA
    cudaMemcpy(A_from_CUDA.get(), cuA, sizeof(float)*N*N, cudaMemcpyDeviceToHost);    // copy from gpu to cpu
    assert(cudaGetLastError() == cudaSuccess);

    // Free cuA, cuB and cuC
    cudaFree(cuA);
    assert(cudaGetLastError() == cudaSuccess);
    cudaFree(cuB);
    assert(cudaGetLastError() == cudaSuccess);
    cudaFree(cuC);
    assert(cudaGetLastError() == cudaSuccess);

    // Check result
    float error = 0;
    for(int i = 0 ; i < N ; ++i){
        for(int j = 0 ; j < N ; ++j){
            error = std::max(error, (A[i*N+j] == 0 ? A_from_CUDA[i*N+j] : std::abs((A_from_CUDA[i*N+j]-A[i*N+j])/A[i*N+j])));
        }
    }

    std::cout << "Error is : " << error << std::endl;

    return 0;
}
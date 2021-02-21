
#include <cuda.h>
#include <limits>

__global__ void sumThemAll(const long int* valuesToSum, long int* currentSum, const long int N){
    // TODO if thread id is less than N, sum the values
    const int i_start = blockIdx.x*blockDim.x + threadIdx.x;

    const int i_inc = gridDim.x*blockDim.x;

    currentSum[i_start] = 0;
    for(int i = i_start ; i < N ; i += i_inc){
            currentSum[i_start] += valuesToSum[i];
    }
}


#include <memory>
#include <iostream>

int main(){
    const int MaxThreadsPerGroup = 10;
    const int MaxGroup = 5;
    const int N = 1000;

    std::unique_ptr<long int[]> values(new long int[N]);

    for(int i = 0 ; i < N ; ++i){
        values[i] = i;
    }
    const long int expectedSum = (N-1)*(N-1+1)/2;

    long int finalSum = -1;
    
    // Malloc and copy
    long int* cuValues;
    cudaMalloc(&cuValues, sizeof(long int)*N);
    assert(cudaGetLastError() == cudaSuccess);
    cudaMemcpy(cuValues, values.get(), sizeof(long int)*N, cudaMemcpyHostToDevice);
    assert(cudaGetLastError() == cudaSuccess);

    long int* cuBuffer;
    cudaMalloc(&cuBuffer, sizeof(long int)*N);
    assert(cudaGetLastError() == cudaSuccess);
    cudaMemset(cuBuffer, 0, sizeof(long int)*N);
    assert(cudaGetLastError() == cudaSuccess);

    // TODO loop to sum the array multiple times until there is only one value
    // 1-diemntional arrays, no need for 2D
    int arraySize = N;
    int blockNb = MaxGroup;
    int threadNb = MaxThreadsPerGroup;

    while (arraySize > 1){

        std::cout << "Run for size: " << arraySize << "  groups: " << blockNb << "  threads: " << threadNb << "  total threads: " << threadNb*blockNb << std::endl;

        // Use maximum number of threads if the size of the array is larger than twice the maxnumber of threads. 
        if (arraySize >= 2*MaxGroup*MaxThreadsPerGroup){
            blockNb = MaxGroup;
            threadNb = MaxThreadsPerGroup;
        }

        // Use a convenient amount of threads (larger (if not equal) to the array size)
        CudaCpu(dim3(blockNb), dim3(threadNb), sumThemAll, cuValues, cuBuffer, arraySize);

        // Copy data from device memory to device memory --> likely inneficient! --> use shared memory
        // cudaMemcpy(cuValues, cuBuffer, sizeof(long int)*N, cudaMemcpyDeviceToDevice); // Copy the buffers

        // Instead of copyind data, swap pointers to save time
        std::swap(cuValues, cuBuffer);                                                   // Swap the buffers

        // The array size for the next iteration step is the total number of threads for this step 
        arraySize = threadNb*blockNb;
        // Try reducing the number of blocks first
        if (blockNb > 1) blockNb = blockNb/2 + blockNb%2;
        // For the next iteration, the total threads should idealy be greater than half the array size
        else threadNb = threadNb/2 + threadNb%2;

    }

    // Get back results
    // The first element in cuValues should contain the final sum
    cudaMemcpy(&finalSum, cuValues, sizeof(long int), cudaMemcpyDeviceToHost);
    assert(cudaGetLastError() == cudaSuccess);

    // Free
    cudaFree(cuValues);
    assert(cudaGetLastError() == cudaSuccess);
    cudaFree(cuBuffer);
    assert(cudaGetLastError() == cudaSuccess);

    
    if(finalSum == expectedSum){
        std::cout << "Correct! Sum found is : " << finalSum << std::endl;    
    }
    else{
        std::cout << "Error! Sum found is : " << finalSum << " should be " << expectedSum << std::endl;       
    }

    return 0;
}
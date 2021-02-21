
__host__ ​ __device__ ​cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor ( int* numBlocks, const void* func, int  blockSize, size_t dynamicSMemSize )
    Returns occupancy for a device function. 
__host__ ​cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags ( int* numBlocks, const void* func, int  blockSize, size_t dynamicSMemSize, unsigned int  flags )
    Returns occupancy for a device function with the specified flags. 

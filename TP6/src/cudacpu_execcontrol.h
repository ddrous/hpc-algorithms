
__host__ ​ __device__ ​cudaError_t cudaFuncGetAttributes ( cudaFuncAttributes* attr, const void* func )
    Find out attributes for a given function. 
__host__ ​cudaError_t cudaFuncSetAttribute ( const void* func, cudaFuncAttribute attr, int  value )
    Set attributes for a given function. 
__host__ ​cudaError_t cudaFuncSetCacheConfig ( const void* func, cudaFuncCache cacheConfig )
    Sets the preferred cache configuration for a device function. 
__host__ ​cudaError_t cudaFuncSetSharedMemConfig ( const void* func, cudaSharedMemConfig config )
    Sets the shared memory configuration for a device function. 
__device__ ​ void* cudaGetParameterBuffer ( size_t alignment, size_t size )
    Obtains a parameter buffer. 
__device__ ​ void* cudaGetParameterBufferV2 ( void* func, dim3 gridDimension, dim3 blockDimension, unsigned int  sharedMemSize )
    Launches a specified kernel. 
__host__ ​cudaError_t cudaLaunchCooperativeKernel ( const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream )
    Launches a device function where thread blocks can cooperate and synchronize as they execute. 
__host__ ​cudaError_t cudaLaunchCooperativeKernelMultiDevice ( cudaLaunchParams* launchParamsList, unsigned int  numDevices, unsigned int  flags = 0 )
    Launches device functions on multiple devices where thread blocks can cooperate and synchronize as they execute. 
__host__ ​cudaError_t cudaLaunchHostFunc ( cudaStream_t stream, cudaHostFn_t fn, void* userData )
    Enqueues a host function call in a stream. 
__host__ ​cudaError_t cudaLaunchKernel ( const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream )
    Launches a device function. 
__host__ ​cudaError_t cudaSetDoubleForDevice ( double* d )
    Converts a double argument to be executed on a device. 
__host__ ​cudaError_t cudaSetDoubleForHost ( double* d )
    Converts a double argument after execution on a device. 

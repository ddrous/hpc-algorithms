__host__ ​cudaError_t cudaEventCreate ( cudaEvent_t* event )
    Creates an event object. 
__host__ ​ __device__ ​cudaError_t cudaEventCreateWithFlags ( cudaEvent_t* event, unsigned int  flags )
    Creates an event object with the specified flags. 
__host__ ​ __device__ ​cudaError_t cudaEventDestroy ( cudaEvent_t event )
    Destroys an event object. 
__host__ ​cudaError_t cudaEventElapsedTime ( float* ms, cudaEvent_t start, cudaEvent_t end )
    Computes the elapsed time between events. 
__host__ ​cudaError_t cudaEventQuery ( cudaEvent_t event )
    Queries an event's status. 
__host__ ​ __device__ ​cudaError_t cudaEventRecord ( cudaEvent_t event, cudaStream_t stream = 0 )
    Records an event. 
__host__ ​cudaError_t cudaEventSynchronize ( cudaEvent_t event ) 

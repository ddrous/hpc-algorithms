__host__ ​cudaError_t cudaStreamAddCallback ( cudaStream_t stream, cudaStreamCallback_t callback, void* userData, unsigned int  flags )
    Add a callback to a compute stream. 
__host__ ​cudaError_t cudaStreamAttachMemAsync ( cudaStream_t stream, void* devPtr, size_t length = 0, unsigned int  flags = cudaMemAttachSingle )
    Attach memory to a stream asynchronously. 
__host__ ​cudaError_t cudaStreamBeginCapture ( cudaStream_t stream, cudaStreamCaptureMode mode )
    Begins graph capture on a stream. 
__host__ ​cudaError_t cudaStreamCreate ( cudaStream_t* pStream )
    Create an asynchronous stream. 
__host__ ​ __device__ ​cudaError_t cudaStreamCreateWithFlags ( cudaStream_t* pStream, unsigned int  flags )
    Create an asynchronous stream. 
__host__ ​cudaError_t cudaStreamCreateWithPriority ( cudaStream_t* pStream, unsigned int  flags, int  priority )
    Create an asynchronous stream with the specified priority. 
__host__ ​ __device__ ​cudaError_t cudaStreamDestroy ( cudaStream_t stream )
    Destroys and cleans up an asynchronous stream. 
__host__ ​cudaError_t cudaStreamEndCapture ( cudaStream_t stream, cudaGraph_t* pGraph )
    Ends capture on a stream, returning the captured graph. 
__host__ ​cudaError_t cudaStreamGetCaptureInfo ( cudaStream_t stream, cudaStreamCaptureStatus ** pCaptureStatus, unsigned long long* pId )
    Query capture status of a stream. 
__host__ ​cudaError_t cudaStreamGetFlags ( cudaStream_t hStream, unsigned int* flags )
    Query the flags of a stream. 
__host__ ​cudaError_t cudaStreamGetPriority ( cudaStream_t hStream, int* priority )
    Query the priority of a stream. 
__host__ ​cudaError_t cudaStreamIsCapturing ( cudaStream_t stream, cudaStreamCaptureStatus ** pCaptureStatus )
    Returns a stream's capture status. 
__host__ ​cudaError_t cudaStreamQuery ( cudaStream_t stream )
    Queries an asynchronous stream for completion status. 
__host__ ​cudaError_t cudaStreamSynchronize ( cudaStream_t stream )
    Waits for stream tasks to complete. 
__host__ ​ __device__ ​cudaError_t cudaStreamWaitEvent ( cudaStream_t stream, cudaEvent_t event, unsigned int  flags )
    Make a compute stream wait on an event. 
__host__ ​cudaError_t cudaThreadExchangeStreamCaptureMode ( cudaStreamCaptureMode ** mode ) 

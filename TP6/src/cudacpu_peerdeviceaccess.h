
__host__ ​cudaError_t cudaDeviceCanAccessPeer ( int* canAccessPeer, int  device, int  peerDevice )
    Queries if a device may directly access a peer device's memory. 
__host__ ​cudaError_t cudaDeviceDisablePeerAccess ( int  peerDevice )
    Disables direct access to memory allocations on a peer device. 
__host__ ​cudaError_t cudaDeviceEnablePeerAccess ( int  peerDevice, unsigned int  flags )
    Enables direct access to memory allocations on a peer device. 

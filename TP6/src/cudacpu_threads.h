__threadfence_block(); 	wait until memory accesses are visible to block
__threadfence();       	wait until memory accesses are visible to block and device
__threadfence_system(); 	wait until memory accesses are visible to block and device and host (2.x)
__syncthreads();

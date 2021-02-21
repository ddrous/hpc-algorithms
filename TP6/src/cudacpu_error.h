#ifndef CUDACPU_ERROR_H
#define CUDACPU_ERROR_H

#include "cudacpu_syntax.h"

enum cudaError{
  cudaSuccess = 0,
  cudaErrorMissingConfiguration,
  cudaErrorMemoryAllocation,
  cudaErrorInitializationError,
  cudaErrorLaunchFailure,
  cudaErrorPriorLaunchFailure,
  cudaErrorLaunchTimeout,
  cudaErrorLaunchOutOfResources,
  cudaErrorInvalidDeviceFunction,
  cudaErrorInvalidConfiguration,
  cudaErrorInvalidDevice,
  cudaErrorInvalidValue,
  cudaErrorInvalidPitchValue,
  cudaErrorInvalidSymbol,
  cudaErrorMapBufferObjectFailed,
  cudaErrorUnmapBufferObjectFailed,
  cudaErrorInvalidHostPointer,
  cudaErrorInvalidDevicePointer,
  cudaErrorInvalidTexture,
  cudaErrorInvalidTextureBinding,
  cudaErrorInvalidChannelDescriptor,
  cudaErrorInvalidMemcpyDirection,
  cudaErrorAddressOfConstant,
  cudaErrorTextureFetchFailed,
  cudaErrorTextureNotBound,
  cudaErrorSynchronizationError,
  cudaErrorInvalidFilterSetting,
  cudaErrorInvalidNormSetting,
  cudaErrorMixedDeviceExecution,
  cudaErrorCudartUnloading,
  cudaErrorUnknown,
  cudaErrorNotYetImplemented,
  cudaErrorMemoryValueTooLarge,
  cudaErrorInvalidResourceHandle,
  cudaErrorNotReady,
  cudaErrorStartupFailure = 0x7f,
  cudaErrorApiFailureBase = 10000
};

typedef enum cudaError cudaError_t;

cudaError_t cudacpu_globalcurrenterror;

__host__ __device__ const char* cudaGetErrorName ( cudaError_t error ){
    switch(error){
    case cudaSuccess: return "cudaSuccess";
    case cudaErrorMissingConfiguration: return "cudaErrorMissingConfiguration";
    case cudaErrorMemoryAllocation: return "cudaErrorMemoryAllocation";
    case cudaErrorInitializationError: return "cudaErrorInitializationError";
    case cudaErrorLaunchFailure: return "cudaErrorLaunchFailure";
    case cudaErrorPriorLaunchFailure: return "cudaErrorPriorLaunchFailure";
    case cudaErrorLaunchTimeout: return "cudaErrorLaunchTimeout";
    case cudaErrorLaunchOutOfResources: return "cudaErrorLaunchOutOfResources";
    case cudaErrorInvalidDeviceFunction: return "cudaErrorInvalidDeviceFunction";
    case cudaErrorInvalidConfiguration: return "cudaErrorInvalidConfiguration";
    case cudaErrorInvalidDevice: return "cudaErrorInvalidDevice";
    case cudaErrorInvalidValue: return "cudaErrorInvalidValue";
    case cudaErrorInvalidPitchValue: return "cudaErrorInvalidPitchValue";
    case cudaErrorInvalidSymbol: return "cudaErrorInvalidSymbol";
    case cudaErrorMapBufferObjectFailed: return "cudaErrorMapBufferObjectFailed";
    case cudaErrorUnmapBufferObjectFailed: return "cudaErrorUnmapBufferObjectFailed";
    case cudaErrorInvalidHostPointer: return "cudaErrorInvalidHostPointer";
    case cudaErrorInvalidDevicePointer: return "cudaErrorInvalidDevicePointer";
    case cudaErrorInvalidTexture: return "cudaErrorInvalidTexture";
    case cudaErrorInvalidTextureBinding: return "cudaErrorInvalidTextureBinding";
    case cudaErrorInvalidChannelDescriptor: return "cudaErrorInvalidChannelDescriptor";
    case cudaErrorInvalidMemcpyDirection: return "cudaErrorInvalidMemcpyDirection";
    case cudaErrorAddressOfConstant: return "cudaErrorAddressOfConstant";
    case cudaErrorTextureFetchFailed: return "cudaErrorTextureFetchFailed";
    case cudaErrorTextureNotBound: return "cudaErrorTextureNotBound";
    case cudaErrorSynchronizationError: return "cudaErrorSynchronizationError";
    case cudaErrorInvalidFilterSetting: return "cudaErrorInvalidFilterSetting";
    case cudaErrorInvalidNormSetting: return "cudaErrorInvalidNormSetting";
    case cudaErrorMixedDeviceExecution: return "cudaErrorMixedDeviceExecution";
    case cudaErrorCudartUnloading: return "cudaErrorCudartUnloading";
    case cudaErrorUnknown: return "cudaErrorUnknown";
    case cudaErrorNotYetImplemented: return "cudaErrorNotYetImplemented";
    case cudaErrorMemoryValueTooLarge: return "cudaErrorMemoryValueTooLarge";
    case cudaErrorInvalidResourceHandle: return "cudaErrorInvalidResourceHandle";
    case cudaErrorNotReady: return "cudaErrorNotReady";
    case cudaErrorStartupFailure: return "cudaErrorStartupFailure";
    case cudaErrorApiFailureBase: return "cudaErrorApiFailureBase";
    }
    return "unknown error code";
}

__host__ __device__ const char* cudaGetErrorString ( cudaError_t error ){
    return cudaGetErrorName(error);
}

__host__ __device__ cudaError_t cudaGetLastError ( void ){
    const cudaError_t currentstate = cudacpu_globalcurrenterror;
    cudacpu_globalcurrenterror = cudaSuccess;
    return currentstate;
}

__host__ __device__ cudaError_t cudaPeekAtLastError ( void ){
    return cudacpu_globalcurrenterror;
}

#endif

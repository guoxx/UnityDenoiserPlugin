#pragma once

#include <nvrhi/nvrhi.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <optix.h>

namespace UnityDenoiserPlugin
{

enum Readback {
    None = 0,
    Albedo = 1,
    Normal = 2,
    Flow = 3,
    Color = 4,
    PreviousOutput = 5,
};

void SignalExternalSemaphore( cudaExternalSemaphore_t extSem, unsigned long long value, cudaStream_t stream );

void WaitExternalSemaphore( cudaExternalSemaphore_t extSem, unsigned long long value, cudaStream_t stream );

inline uint32_t DivideRoundUp( uint32_t a, uint32_t b )
{
    return ( a + b - 1 ) / b;
}

void LogMessage( const char* msg );
void LogWarning( const char* msg );
void LogError( const char* msg );

}
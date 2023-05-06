#include "Utils.h"

extern IUnityLog* UnityLogger();

namespace UnityDenoiserPlugin
{

void SignalExternalSemaphore( cudaExternalSemaphore_t extSem, unsigned long long value, cudaStream_t stream )
{
    cudaExternalSemaphoreSignalParams params = {};
    memset( &params, 0, sizeof( params ) );
    params.params.fence.value = value;

    cudaSignalExternalSemaphoresAsync( &extSem, &params, 1, stream );
}

void WaitExternalSemaphore( cudaExternalSemaphore_t extSem, unsigned long long value, cudaStream_t stream )
{
    cudaExternalSemaphoreWaitParams params = {};
    memset( &params, 0, sizeof( params ) );
    params.params.fence.value = value;

    cudaWaitExternalSemaphoresAsync( &extSem, &params, 1, stream );
}

void LogMessage( const char* msg )
{
    UNITY_LOG( UnityLogger(), msg );
}

void LogWarning( const char* msg )
{
    UNITY_LOG_WARNING( UnityLogger(), msg );
}

void LogError( const char* msg )
{
    UNITY_LOG_ERROR( UnityLogger(), msg );
}

}
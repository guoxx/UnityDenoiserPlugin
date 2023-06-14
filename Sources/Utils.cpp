#include "Utils.h"

extern IUnityLog* UnityLogger();

namespace UnityDenoiserPlugin
{

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
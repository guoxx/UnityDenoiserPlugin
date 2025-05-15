#include "Utils.h"
#include <cstdarg>
#include <cstdio>

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

void LogMessageFormat( const char* format, ... )
{
    va_list args;
    va_start(args, format);
    char buffer[1024];
    vsnprintf(buffer, sizeof(buffer), format, args);
    LogMessage(buffer);
    va_end(args);
}

void LogWarningFormat( const char* format, ... )
{
    va_list args;
    va_start(args, format);
    char buffer[1024];
    vsnprintf(buffer, sizeof(buffer), format, args);
    LogWarning(buffer);
    va_end(args);
}

void LogErrorFormat( const char* format, ... )
{
    va_list args;
    va_start(args, format);
    char buffer[1024];
    vsnprintf(buffer, sizeof(buffer), format, args);
    LogError(buffer);
    va_end(args);
}

}
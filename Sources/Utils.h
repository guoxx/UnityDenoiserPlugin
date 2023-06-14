#pragma once

namespace UnityDenoiserPlugin
{

inline uint32_t DivideRoundUp( uint32_t a, uint32_t b )
{
    return ( a + b - 1 ) / b;
}

void LogMessage( const char* msg );
void LogWarning( const char* msg );
void LogError( const char* msg );

}
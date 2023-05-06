#include "SimpleLogger.h"
#include <cstdio>
#include <iostream>
#include <vector>

namespace oidn
{

using namespace nvneural;

constexpr uint32_t kLogLevelError = 0;
constexpr uint32_t kLogLevelWarning = 1;
constexpr uint32_t kLogLevelInfo = 2;

SimpleLogger::SimpleLogger(VerbosityLevel verbosity)
    : m_verbosity(verbosity)
{
}

void SimpleLogger::setVerbosity(VerbosityLevel verbosity)
{
    m_verbosity = verbosity;
}

ILogger::VerbosityLevel SimpleLogger::verbosity() const noexcept
{
    return m_verbosity;
}

NeuralResult SimpleLogger::log(VerbosityLevel verbosity, const char* format, ...) noexcept
{
    if (verbosity > m_verbosity)
    {
        return NeuralResult::Success;
    }

    va_list args;
    va_start(args, format);
    const NeuralResult logResult = logImpl(kLogLevelInfo, format, args);
    va_end(args);
    return logResult;
}

NeuralResult SimpleLogger::logWarning(VerbosityLevel verbosity, const char* format, ...) noexcept
{
    if (verbosity > m_verbosity)
    {
        return NeuralResult::Success;
    }

    va_list args;
    va_start(args, format);
    const NeuralResult logResult = logImpl(kLogLevelWarning, format, args);
    va_end(args);
    return logResult;
}

NeuralResult SimpleLogger::logError(VerbosityLevel verbosity, const char* format, ...) noexcept
{
    if (verbosity > m_verbosity)
    {
        return NeuralResult::Success;
    }

    va_list args;
    va_start(args, format);
    const NeuralResult logResult = logImpl(kLogLevelError, format, args);
    va_end(args);
    return logResult;
}

NeuralResult SimpleLogger::logImpl(uint32_t level, const char* format, va_list formatArgs) noexcept
{
    static char shortLogBuffer[8192];
    static const int ShortLogBufferSize = static_cast<int>(sizeof(shortLogBuffer));

    try
    {
        va_list sizingArgs;
        va_copy(sizingArgs, formatArgs);

        const int numBytes = 1 + vsnprintf(nullptr, 0, format, sizingArgs); // count the null terminator

        va_end(sizingArgs);

        if (numBytes <= 0)
        {
            return NeuralResult::Failure; // vsnprintf failed
        }

        if (numBytes <= ShortLogBufferSize)
        {
            const int bytesFormatted = 1 + vsnprintf(shortLogBuffer, ShortLogBufferSize, format, formatArgs);
            if (bytesFormatted <= 0)
            {
                return NeuralResult::Failure;
            }

            const NeuralResult writeResult = writeFormattedLog(level, shortLogBuffer);
            return writeResult;
        }
        else
        {
            std::vector<char> dynamicBuffer(numBytes);
            const int bytesFormatted = 1 + vsnprintf(dynamicBuffer.data(), dynamicBuffer.size(), format, formatArgs);
            if (bytesFormatted <= 0)
            {
                return NeuralResult::Failure;
            }
            const NeuralResult writeResult = writeFormattedLog(level, dynamicBuffer.data());
            return writeResult;
        }
    }
    catch (...)
    {
        return NeuralResult::Failure;
    }
}

NeuralResult SimpleLogger::writeFormattedLog(uint32_t level, char* formattedMessage)
{
    if (m_logCallback)
    {
        m_logCallback(level, formattedMessage);
    }
    else
    {
        if (level == kLogLevelError)
            std::cerr << formattedMessage << std::endl;
        else
            std::cout << formattedMessage << std::endl;
    }
    return NeuralResult::Success;
}

}
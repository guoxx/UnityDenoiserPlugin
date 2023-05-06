/*
* SPDX-FileCopyrightText: Copyright (c) 2020-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
* SPDX-License-Identifier: MIT
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/
 
#pragma once

#include <nvneural/CoreTypes.h>
#include <nvneural/CoreHelpers.h>

#include <cstdarg>
#include <iostream>

namespace oidn {

typedef void (*LogCallback)(unsigned int level, const char *message);

/// SimpleLogger is a basic ILogger implementation that displays to stdout.
///
/// Logs are sent unaltered, warnings are prefixed with "WRN: ", and errors are
/// prefixed with "ERR: ". ConverenceNG uses SimpleLogger for the majority of its
/// command-line output.
class SimpleLogger : public nvneural::refobj::RefObjectBase<
    nvneural::refobj::Implements<nvneural::ILogger>>
{
public:
    /// Creates a SimpleLogger with a default verbosity level.
    ///
    /// \param verbosity Default verbosity level to assign
    explicit SimpleLogger(VerbosityLevel verbosity);

    /// Sets the new verbosity threshold for the logger.
    ///
    /// Logs whose verbosity levels exceed the threshold are silently ignored.
    ///
    /// \param verbosity New verbosity level to assign
    void setVerbosity(VerbosityLevel verbosity);

    void setLogCallback(LogCallback callback) { m_logCallback = callback; }

    // ILogger implementations

    /// \copydoc nvneural::ILogger::verbosity
    VerbosityLevel verbosity() const noexcept final;

    /// \copydoc nvneural::ILogger::log
    nvneural::NeuralResult log(VerbosityLevel verbosity, const char* format, ...) noexcept final;

    /// \copydoc nvneural::ILogger::logWarning
    nvneural::NeuralResult logWarning(VerbosityLevel verbosity, const char* format, ...) noexcept final;

    /// \copydoc nvneural::ILogger::logError
    nvneural::NeuralResult logError(VerbosityLevel verbosity, const char* format, ...) noexcept final;

private:
    LogCallback m_logCallback = nullptr;

    VerbosityLevel m_verbosity = 0;

    nvneural::NeuralResult logImpl(uint32_t level, const char* format, va_list formatArgs) noexcept;

    nvneural::NeuralResult writeFormattedLog(uint32_t level, char* formattedMessage);
};

}

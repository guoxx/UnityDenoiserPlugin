#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

# Locate the OptiX distribution.  Search relative to the SDK first, then look in the system.

# Our initial guess will be within the SDK.
if (DEFINED ENV{OptiX_INSTALL_DIR})
  message(STATUS "Detected the OptiX_INSTALL_DIR env variable (pointing to $ENV{OptiX_INSTALL_DIR}; going to use this for finding optix.h")
  find_path(OptiX_INSTALL_DIR NAMES include/optix.h PATHS $ENV{OptiX_INSTALL_DIR})
else()
  find_path(searched_OptiX_INSTALL_DIR
    NAME include/optix.h
    PATHS
    "C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.0.0"
    "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.7.0"
    "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.6.0"
    "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.5.0"
    "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.4.0"
    "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.3.0"
    "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.2.0"
    "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.1.0"
    "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.0.0"
    "C:/ProgramData/NVIDIA Corporation/OptiX SDK 6.5.0"
    "C:/ProgramData/NVIDIA Corporation/OptiX SDK 6.0.0"
    "C:/ProgramData/NVIDIA Corporation/OptiX SDK 5.1.1"
    "C:/ProgramData/NVIDIA Corporation/OptiX SDK 5.1.0"
    "C:/ProgramData/NVIDIA Corporation/OptiX SDK 5.0.1"
    "C:/ProgramData/NVIDIA Corporation/OptiX SDK 5.0.0"
    "C:/ProgramData/NVIDIA Corporation/OptiX SDK *"
  )

  mark_as_advanced(searched_OptiX_INSTALL_DIR)
  set(OptiX_INSTALL_DIR ${searched_OptiX_INSTALL_DIR} CACHE PATH "Path to OptiX installed location.")
endif()

# The distribution contains only 64 bit libraries.  Error when we have been mis-configured.
if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
  if(WIN32)
    message(SEND_ERROR "Make sure when selecting the generator, you select one with Win64 or x64.")
  endif()
  message(FATAL_ERROR "OptiX only supports builds configured for 64 bits.")
endif()

# search path based on the bit-ness of the build.  (i.e. 64: bin64, lib64; 32:
# bin, lib).  Note that on Mac, the OptiX library is a universal binary, so we
# only need to look in lib and not lib64 for 64 bit builds.
if(NOT APPLE)
  set(bit_dest "64")
else()
  set(bit_dest "")
endif()

# Include
find_path(OptiX_INCLUDE
  NAMES optix.h
  PATHS "${OptiX_INSTALL_DIR}/include"
  NO_DEFAULT_PATH
  )
find_path(OptiX_INCLUDE
  NAMES optix.h
  )

# Check to make sure we found what we were looking for
function(OptiX_report_error error_message required component )
  if(DEFINED OptiX_FIND_REQUIRED_${component} AND NOT OptiX_FIND_REQUIRED_${component})
    set(required FALSE)
  endif()
  if(OptiX_FIND_REQUIRED AND required)
    message(FATAL_ERROR "${error_message}  Please locate before proceeding.")
  else()
    if(NOT OptiX_FIND_QUIETLY)
      message(STATUS "${error_message}")
    endif(NOT OptiX_FIND_QUIETLY)
  endif()
endfunction()

if(NOT OptiX_INCLUDE)
  OptiX_report_error("OptiX headers (optix.h and friends) not found." TRUE headers )
endif()


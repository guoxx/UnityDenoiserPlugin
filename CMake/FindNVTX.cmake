#
#  Copyright (c) 2021 NVIDIA Corporation.  All rights reserved.
#
#  NVIDIA Corporation and its licensors retain all intellectual property and proprietary
#  rights in and to this software, related documentation and any modifications thereto.
#  Any use, reproduction, disclosure or distribution of this software and related
#  documentation without an express license agreement from NVIDIA Corporation is strictly
#  prohibited.
#
#  TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
#  AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
#  INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#  PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
#  SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
#  LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
#  BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
#  INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
#  SUCH DAMAGES
#

# Output variables:
#   NVTX_FOUND
#   NVTX_INCLUDE_DIR

if( WIN32 )
    # On Windows, the NVTX headers are in the include directory of the CUDA Toolkit
    find_path( NVTX_INCLUDE_DIR
        NAMES device_functions.h
        PATHS
            ${CUDA_TOOLKIT_ROOT_DIR}
            ENV CUDA_PATH
            ENV CUDA_INC_PATH
        PATH_SUFFIXES include
        NO_DEFAULT_PATH
        )
elseif( UNIX )
    # On Linux, the NVTX headers are in a separate 'targets' directory
    find_path( NVTX_INCLUDE_DIR
        NAMES nvToolsExt.h
        PATHS
            ${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/include
            ENV CUDA_PATH
            ENV CUDA_INC_PATH
        PATH_SUFFIXES include
        NO_DEFAULT_PATH
        )
endif()

if( NVTX_INCLUDE_DIR )
    set( NVTX_FOUND TRUE )
endif()

mark_as_advanced( NVTX_INCLUDE_DIR )

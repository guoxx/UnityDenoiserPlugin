# This file was automatically generated, but should be customized/replaced to meet your needs.

cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)
project(OIDN CXX CUDA)

find_package(CUDAToolkit)

# Try to find NvNeural. Search the environment variable NVNEURAL_SDK_PATH, the local NvNeuralSDK/ if it exists, and
# a path based off the ConverenceNG install used to generate this code.
#
# Note that this variable is stored in CMake's cache once the SDK is located; you will need to clear the cache if
# you intend to change $NVNEURAL_SDK_PATH and compile against a different SDK.
if(NOT DEFINED CACHE{NVNEURAL_SDK_PATH})
    find_path(
        NVNEURAL_SDK_PATH
        NAMES NvNeuralSDK.cmake
        PATHS ENV NVNEURAL_SDK_PATH NvNeuralSDK "C:/Program Files/NVIDIA Corporation/Nsight DL Designer 2022.2.0/host/windows-desktop-dl-x64/NvNeuralSDK"
        DOC "Path to NvNeural SDK"
        REQUIRED)
    if(NVNEURAL_SDK_PATH)
        message(STATUS "Found NvNeural SDK: ${NVNEURAL_SDK_PATH}")
    else()
        message(FATAL_ERROR "NvNeural SDK not found.")
    endif()
endif()

# Out-of-source add_subdirectory() requires a relative path to store generated binaries
add_subdirectory(${NVNEURAL_SDK_PATH} NvNeuralSDK)

add_executable(OIDN
    BuildNetwork_OIDN.h # for VS generators
    BuildNetwork_OIDN.cpp
    main.cpp)
target_link_libraries(OIDN PRIVATE NvNeural::Host)

install(TARGETS OIDN DESTINATION .)
NvNeural_InstallStandardPlugins(DESTINATION .)

# Deploy the toolkit's CUDA runtime compiler alongside the app.
# Feel free to customize the list if you add new CUDA dependencies or change toolkits.
# NvRTC in 11.x uses the 11.2 version for its outer SONAME but a toolkit-versioned suffix for nvrtc-builtins.
if("${CMAKE_SYSTEM_NAME}" STREQUAL "Windows")
    install(
        FILES
            ${CUDAToolkit_BIN_DIR}/nvrtc64_112_0.dll
            ${CUDAToolkit_BIN_DIR}/nvrtc-builtins64_${CUDAToolkit_VERSION_MAJOR}${CUDAToolkit_VERSION_MINOR}.dll
        DESTINATION .)
else()
    # Assume non-Windows platforms all use versioned .so names
    install(
        FILES
            ${CUDAToolkit_LIBRARY_DIR}/libnvrtc.so.11.2
            ${CUDAToolkit_LIBRARY_DIR}/libnvrtc-builtins.so
            ${CUDAToolkit_LIBRARY_DIR}/libnvrtc-builtins.so.${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR}
        DESTINATION .
    )
endif()

# Deploy cuDNN alongside the app, since it's necessary for the convolution plugin.
# By default, cuDNN should be installed into your CUDA toolkit directory after download.
# See https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html for details.
#
# If you have installed cuDNN to a different location, set the environment variable
# CUDNN_SDK_PATH or CUDNN_PATH to the base SDK path (i.e., the level above bin/) before configuring.
#
# Note: If you don't need cuDNN for your application, feel free to remove this section!
set(CudnnLibraryNames "cudnn" "cudnn_adv_infer" "cudnn_cnn_infer" "cudnn_ops_infer")

# Search if all of the cuDNN libraries can be found on user's system.
set(FoundCudnnOnUserSystem ON)
foreach(CudnnLibrary ${CudnnLibraryNames})
    find_library(
        CUDNN_${CudnnLibrary}_LIBRARY
        NAMES ${CudnnLibrary}
        PATHS ENV CUDNN_SDK_PATH CUDNN_PATH ${CUDAToolkit_LIBRARY_DIR}) # try %CUDNN_SDK_PATH%, then %CUDNN_PATH%, then the CUDA path
    if (NOT CUDNN_${CudnnLibrary}_LIBRARY)
        set(FoundCudnnOnUserSystem OFF)
    endif()
endforeach()

foreach(CudnnLibrary ${CudnnLibraryNames})
    # LibName is the filename of the library we're looking for
    if (WIN32)
        set(LibName "${CMAKE_SHARED_LIBRARY_PREFIX}${CudnnLibrary}64_8${CMAKE_SHARED_LIBRARY_SUFFIX}")   # (e.g. "cudnn64_8.dll")
    else()
        set(LibName "${CMAKE_SHARED_LIBRARY_PREFIX}${CudnnLibrary}${CMAKE_SHARED_LIBRARY_SUFFIX}.8")  # (e.g. "libcudnn.so.8")
    endif()
    if (FoundCudnnOnUserSystem) # Found the library on user's system
        if (WIN32)
            find_file(
                CUDNN_${CudnnLibrary}_FILE
                NAMES ${LibName}
                HINTS ${CUDNN_${CudnnLibrary}_LIBRARY}/../../../bin)
            install(FILES ${CUDNN_${CudnnLibrary}_FILE} DESTINATION .)
        else()
            install(FILES ${CUDNN_${CudnnLibrary}_LIBRARY} DESTINATION .)
        endif()
    else()  # Cannot find the library on user's system
        install(FILES ${NVNEURAL_SDK_PATH}/Bin/${LibName} DESTINATION .)
    endif()
endforeach()


# Deploy cuBLAS alongside the app, since it's necessary for the fully-connected plugin and
# the gram matrix plugin.
# By default, cuBLAS should be installed into your CUDA toolkit directory after download.
#
# If you have installed cuBLAS to a different location, set the environment variable
# CUBLAS_SDK_PATH or CUBLAS_PATH to the base SDK path (i.e., the level above bin/) before configuring.
#
# Note: If you don't need cuBLAS for your application, feel free to remove this section!

set(CublasLibraryNames "cublas" "cublasLt")

# First search if all of the cuBLAS libraries are in the CUDA toolkit directory.
set(FoundCublasInCTKDir ON)
foreach(CublasLibrary ${CublasLibraryNames})
    if (NOT TARGET CUDA::${CublasLibrary})
        set(FoundCublasInCTKDir OFF)
    endif()
endforeach()

# If all of the cuBLAS libraries cannot be found in the CUDA toolkit directory, we search for the directories pointed 
# by CUBLAS_SDK_PATH and CUBLAS_PATH and see if all of the cuBLAS libraries are in either of them.
if (NOT FoundCublasInCTKDir)
    set(FoundCublasInNonCTKDir ON)
    foreach(CublasLibrary ${CublasLibraryNames})
        find_library(
            CUBLAS_${CublasLibrary}_LIBRARY
            NAMES ${CublasLibrary}
            PATHS ENV CUBLAS_SDK_PATH CUBLAS_PATH) # try %CUBLAS_SDK_PATH%, then %CUBLAS_PATH%
        if (NOT CUBLAS_${CublasLibrary}_LIBRARY)
            set(FoundCublasInNonCTKDir OFF)
        endif()
    endforeach()
endif()

foreach(CublasLibrary ${CublasLibraryNames})
    # LibName is the filename of the library we're looking for
    if (WIN32)
        set(LibName "${CMAKE_SHARED_LIBRARY_PREFIX}${CublasLibrary}64_11${CMAKE_SHARED_LIBRARY_SUFFIX}")   # (e.g. "cublas64_11.dll")
    else()
        set(LibName "${CMAKE_SHARED_LIBRARY_PREFIX}${CublasLibrary}${CMAKE_SHARED_LIBRARY_SUFFIX}.11")  # (e.g. "libcublas.so.11")
    endif()
    if (FoundCublasInCTKDir)    # Found the library in the CUDA toolkit directory on user's system
        if (WIN32)
            install(FILES ${CUDAToolkit_BIN_DIR}/${LibName} DESTINATION .)
        else()
            install(FILES ${CUDAToolkit_LIBRARY_DIR}/${LibName} DESTINATION .)
        endif()
    elseif(FoundCublasInNonCTKDir)    # Found the library in the directories pointed by CUBLAS_SDK_PATH or CUBLAS_PATH on user's system
        if (WIN32)
            find_file(
                CUBLAS_${CublasLibrary}_FILE
                NAMES ${LibName}
                HINTS ${CUBLAS_${CublasLibrary}_LIBRARY}/../../../bin)
            install(FILES ${CUBLAS_${CublasLibrary}_FILE} DESTINATION .)
        else()
            install(FILES ${CUBLAS_${CublasLibrary}_LIBRARY} DESTINATION .)
        endif()
    else()  # Cannot find the library on user's system
        install(FILES ${NVNEURAL_SDK_PATH}/Bin/${LibName} DESTINATION .)
    endif()
endforeach()



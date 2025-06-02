cmake_minimum_required(VERSION 3.12)
project(UnityDenoiserPlugin)
set(target_name "UnityDenoiserPlugin")

set(CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

# Find the OptiX library
find_package(OptiX REQUIRED)
find_package(CUDA REQUIRED)

# DLSS SDK Path
set(DLSS_SDK_PATH ${CMAKE_CURRENT_SOURCE_DIR}/Externals/DLSS)
message(STATUS "Using DLSS SDK from: ${DLSS_SDK_PATH}")

# Add your project's source files here
file(GLOB SOURCE_FILES Sources/*.cpp)

add_library(${target_name} SHARED ${SOURCE_FILES})

# Add include directories
target_include_directories(${target_name} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/Externals/UnityPluginAPI
    ${OptiX_INCLUDE}
    ${CUDA_INCLUDE_DIRS}
    ${DLSS_SDK_PATH}/include # Path to DLSS SDK headers
    )

# pch file
target_precompile_headers(${target_name} PRIVATE Sources/pch.h)

# Link against the necessary OptiX libraries
target_link_libraries(${target_name} ${OptiX_LIBRARIES} ${CUDA_LIBRARIES})

# Link DLSS library
# Fixed to use proper debug/release variants that match runtime libraries
set(DLSS_LIB_DIR ${DLSS_SDK_PATH}/lib/Windows_x86_64/x64 CACHE PATH "Directory containing DLSS .lib files")

# Debug libraries with debugging iterator and MDd runtime
set(DLSS_DEBUG_LIB_NAME "nvsdk_ngx_d_dbg.lib" CACHE STRING "Filename of the DLSS debug .lib file")
# Release libraries with MD runtime
set(DLSS_RELEASE_LIB_NAME "nvsdk_ngx_d.lib" CACHE STRING "Filename of the DLSS release .lib file")

message(STATUS "DLSS: Linking Debug Lib: ${DLSS_LIB_DIR}/${DLSS_DEBUG_LIB_NAME}")
message(STATUS "DLSS: Linking Release Lib: ${DLSS_LIB_DIR}/${DLSS_RELEASE_LIB_NAME}")

target_link_libraries(${target_name}
    $<$<CONFIG:Debug>:${DLSS_LIB_DIR}/${DLSS_DEBUG_LIB_NAME}>
    $<$<NOT:$<CONFIG:Debug>>:${DLSS_LIB_DIR}/${DLSS_RELEASE_LIB_NAME}>
)

# OIDN
set(OIDN_DEVICE_CPU OFF)
set(OIDN_DEVICE_CUDA ON)
set(OIDN_STATIC_LIB ON)
set(OIDN_APPS OFF)
add_subdirectory(Externals/oidn)
target_include_directories(${target_name} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/Externals/oidn/include)
add_dependencies(${target_name} OpenImageDenoise)
target_link_libraries(${target_name} OpenImageDenoise)

# ----------------------------------------------------------------------------
# Post-build step: copy OIDN runtime libraries (core & device) next to the
# UnityDenoiserPlugin DLL. The file names depend on the configurable
# OIDN_LIBRARY_NAME and the enabled devices, therefore we rely on CMake target
# information instead of hard-coding file names.
# ----------------------------------------------------------------------------

# Ensure relevant device modules are built before the copy step.
if (TARGET OpenImageDenoise_device_cuda)
    add_dependencies(${target_name} OpenImageDenoise_device_cuda)
endif()

if (TARGET OpenImageDenoise_device_cpu)
    add_dependencies(${target_name} OpenImageDenoise_device_cpu)
endif()

if (TARGET OpenImageDenoise_device_sycl)
    add_dependencies(${target_name} OpenImageDenoise_device_sycl)
endif()

if (TARGET OpenImageDenoise_device_hip)
    add_dependencies(${target_name} OpenImageDenoise_device_hip)
endif()

if (TARGET OpenImageDenoise_device_metal)
    add_dependencies(${target_name} OpenImageDenoise_device_metal)
endif()

# ------------------------------------------------------------------
# Copy *all* files from the directory where the OIDN core DLL lands.
# The external device libraries are built into the same folder, so a
# single directory copy is robust and avoids the generator-expression
# issue with ExternalProject targets that are not real libraries.
# ------------------------------------------------------------------

add_custom_command(TARGET ${target_name} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        $<TARGET_FILE_DIR:OpenImageDenoise_core>
        $<TARGET_FILE_DIR:${target_name}>
    COMMENT "Copying OIDN runtime binaries to plugin output directory"
)

# Post-build step: copy DLSS runtime libraries

# Define DLSS DLL source directories
set(DLSS_DLL_DEV_DIR ${DLSS_SDK_PATH}/lib/Windows_x86_64/dev)
set(DLSS_DLL_REL_DIR ${DLSS_SDK_PATH}/lib/Windows_x86_64/rel)

# Option to force using the debug DLLs in release builds for troubleshooting
option(DLSS_FORCE_DEBUG_DLL_IN_RELEASE "Force copy of debug DLSS DLLs in Release builds" OFF)

# Determine which directory to use for non-Debug configurations based on the option
if(DLSS_FORCE_DEBUG_DLL_IN_RELEASE)
    set(DLSS_RUNTIME_DLL_PATH ${DLSS_DLL_DEV_DIR})
    message(STATUS "DLSS: DLSS_FORCE_DEBUG_DLL_IN_RELEASE is ON. Non-Debug builds will use DLLs from ${DLSS_RUNTIME_DLL_PATH}.")
else()
    set(DLSS_RUNTIME_DLL_PATH ${DLSS_DLL_REL_DIR})
endif()

# Copy all DLL files from the appropriate DLSS directory
# Debug: copy from dev directory
# Release: copy from rel directory (or dev if DLSS_FORCE_DEBUG_DLL_IN_RELEASE is ON)
add_custom_command(TARGET ${target_name} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        # Copy all files from the selected directory
        "$<$<CONFIG:Debug>:${DLSS_DLL_DEV_DIR}>$<$<NOT:$<CONFIG:Debug>>:${DLSS_RUNTIME_DLL_PATH}>"
        # Destination directory
        "$<TARGET_FILE_DIR:${target_name}>"
    COMMENT "Copying DLSS directory to plugin output directory from: $<$<CONFIG:Debug>:${DLSS_DLL_DEV_DIR}>$<$<NOT:$<CONFIG:Debug>>:${DLSS_RUNTIME_DLL_PATH}>"
)


cmake_minimum_required(VERSION 3.12)
project(UnityDenoiserPlugin)
set(target_name "UnityDenoiserPlugin")

set(CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

# Find the OptiX library
find_package(OptiX REQUIRED)
find_package(CUDA REQUIRED)

# Add your project's source files here
file(GLOB SOURCE_FILES Sources/*.cpp)

add_library(${target_name} SHARED ${SOURCE_FILES})

# Add include directories
target_include_directories(${target_name} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/Externals/UnityPluginAPI
    ${OptiX_INCLUDE}
    ${CUDA_INCLUDE_DIRS}
    )

# pch file
target_precompile_headers(${target_name} PRIVATE Sources/pch.h)

# Link against the necessary OptiX libraries
target_link_libraries(${target_name} ${OptiX_LIBRARIES} ${CUDA_LIBRARIES})

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

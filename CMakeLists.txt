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
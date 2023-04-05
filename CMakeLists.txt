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

set(NVRHI_INSTALL OFF)
set(NVRHI_WITH_SHADER_COMPILER OFF)
set(NVRHI_WITH_VULKAN OFF)
set(NVRHI_WITH_DX11 OFF)
set(NVRHI_WITH_DX12 ON)
add_subdirectory(Externals/NVRHI)

add_dependencies(${target_name} nvrhi nvrhi_d3d12)

# Add include directories
target_include_directories(${target_name} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/Externals/UnityPluginAPI
    ${CMAKE_CURRENT_SOURCE_DIR}/Externals/nvrhi/include
    ${OptiX_INCLUDE}
    ${CUDA_INCLUDE_DIRS}
    )

# pch file
target_precompile_headers(${target_name} PRIVATE Sources/pch.h)

# Link against the necessary OptiX libraries
target_link_libraries(${target_name} ${OptiX_LIBRARIES} ${CUDA_LIBRARIES} nvrhi nvrhi_d3d12)

# Build HLSL shaders
file(GLOB HLSL_SHADER_FILES Sources/Shaders/*.cs.hlsl)

set_source_files_properties(${HLSL_SHADER_FILES} PROPERTIES ShaderType "cs")
set_source_files_properties(${HLSL_SHADER_FILES} PROPERTIES ShaderModel "6_0")

foreach(FILE ${HLSL_SHADER_FILES})
  get_filename_component(FILE_WE ${FILE} NAME_WE)
  get_filename_component(FILE_DIR ${FILE} DIRECTORY)
  get_source_file_property(shadertype ${FILE} ShaderType)
  get_source_file_property(shadermodel ${FILE} ShaderModel)
  add_custom_command(TARGET ${target_name} PRE_BUILD
                     COMMAND dxc.exe /nologo /Emain /T${shadertype}_${shadermodel} $<IF:$<CONFIG:DEBUG>,/Od,/O3> /Zi /Fh ${FILE_DIR}/${FILE_WE}.h /Vn g_${FILE_WE}_ByteCode /Fd ${CMAKE_BINARY_DIR}/${FILE_WE}.pdb ${FILE}
                     MAIN_DEPENDENCY ${FILE}
                     COMMENT "HLSL ${FILE}"
                     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                     VERBATIM)
endforeach(FILE)
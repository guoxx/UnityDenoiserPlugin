
#
#  Copyright (c) 2008 - 2021 NVIDIA Corporation.  All rights reserved.
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

# Appends VAL to the string contained in STR
MACRO(APPEND_TO_STRING STR VAL)
  if (NOT "${ARGN}" STREQUAL "")
    message(SEND_ERROR "APPEND_TO_STRING takes only a single argument to append.  Offending args: ${ARGN}")
  endif()
  # You need to double ${} STR to get the value.  The first one gets
  # the variable, the second one gets the value.
  if (${STR})
    set(${STR} "${${STR}} ${VAL}")
  else()
    set(${STR} "${VAL}")
  endif()
ENDMACRO(APPEND_TO_STRING)

# Prepends VAL to the string contained in STR
MACRO(PREPEND_TO_STRING STR VAL)
  if (NOT "${ARGN}" STREQUAL "")
    message(SEND_ERROR "PREPEND_TO_STRING takes only a single argument to append.  Offending args: ${ARGN}")
  endif()
  # You need to double ${} STR to get the value.  The first one gets
  # the variable, the second one gets the value.
  if (${STR})
    set(${STR} "${VAL} ${${STR}}")
  else()
    set(${STR} "${VAL}")
  endif()
ENDMACRO(PREPEND_TO_STRING)

# Prepends a prefix to items in a list and appends the result to list_out
macro( prepend list_out prefix )
  set( _results )
  foreach( str ${ARGN} )
    list( APPEND _results "${prefix}${str}" )
  endforeach()
  list( APPEND ${list_out} ${_results} )
endmacro()


#################################################################
#  FORCE_ADD_FLAGS(parameter flags)
#
# This will add arguments not found in ${parameter} to the end.  It
# does not attempt to remove duplicate arguments already existing in
# ${parameter}.
#################################################################
MACRO(FORCE_ADD_FLAGS parameter)
  # Create a separated list of the arguments to loop over
  SET(p_list ${${parameter}})
  SEPARATE_ARGUMENTS(p_list)
  # Make a copy of the current arguments in ${parameter}
  SET(new_parameter ${${parameter}})
  # Now loop over each required argument and see if it is in our
  # current list of arguments.
  FOREACH(required_arg ${ARGN})
    # This helps when we get arguments to the function that are
    # grouped as a string:
    #
    # ["-msse -msse2"]  instead of [-msse -msse2]
    SET(TMP ${required_arg}) #elsewise the Seperate command doesn't work)
    SEPARATE_ARGUMENTS(TMP)
    FOREACH(option ${TMP})
      # Look for the required argument in our list of existing arguments
      SET(found FALSE)
      FOREACH(p_arg ${p_list})
        IF (${p_arg} STREQUAL ${option})
          SET(found TRUE)
        ENDIF (${p_arg} STREQUAL ${option})
      ENDFOREACH(p_arg)
      IF(NOT found)
        # The required argument wasn't found, so we need to add it in.
        SET(new_parameter "${new_parameter} ${option}")
      ENDIF(NOT found)
    ENDFOREACH(option ${TMP})
  ENDFOREACH(required_arg ${ARGN})
  SET(${parameter} ${new_parameter} CACHE STRING "" FORCE)
ENDMACRO(FORCE_ADD_FLAGS)

# This MACRO is designed to set variables to default values only on
# the first configure.  Subsequent configures will produce no ops.
MACRO(FIRST_TIME_SET VARIABLE VALUE TYPE COMMENT)
  IF(NOT PASSED_FIRST_CONFIGURE)
    SET(${VARIABLE} ${VALUE} CACHE ${TYPE} ${COMMENT} FORCE)
  ENDIF(NOT PASSED_FIRST_CONFIGURE)
ENDMACRO(FIRST_TIME_SET)

MACRO(FIRST_TIME_MESSAGE)
  IF(NOT PASSED_FIRST_CONFIGURE)
    MESSAGE(${ARGV})
  ENDIF(NOT PASSED_FIRST_CONFIGURE)  
ENDMACRO(FIRST_TIME_MESSAGE)

# Used by ll_to_cpp and bc_to_cpp
find_file(bin2cpp_cmake bin2cpp.cmake ${CMAKE_MODULE_PATH} )
set(bin2cpp_cmake "${bin2cpp_cmake}" CACHE INTERNAL "Path to internal bin2cpp.cmake" FORCE)

# Converts input_ll file to llvm bytecode encoded as a string in the outputSource file
# defined with the export symbol provided.
function(ll_to_cpp input outputSource outputInclude exportSymbol)
  # message("input = ${input}")
  # message("outputSource = ${outputSource}")
  # message("outputInclude = ${outputInclude}")
  # message("exportSymbol = ${exportSymbol}")
  get_filename_component(outputABS  "${outputSource}" ABSOLUTE )
  get_filename_component(outputDir  "${outputSource}" PATH )
  get_filename_component(outputName "${outputSource}" NAME )
  file(RELATIVE_PATH outputRelPath  "${CMAKE_BINARY_DIR}" "${outputDir}")
  
  set(bc_filename "${outputName}.tmp.bc")

  # Generate header file (configure time)
  include(bin2cpp)
  bin2h(${outputInclude} ${exportSymbol} "${bc_filename}")
  
  # Convert ll to byte code
  add_custom_command(
    OUTPUT ${outputSource}

    # convert ll to bc
    COMMAND ${LLVM_llvm-as} "${input}" -o "${bc_filename}"
    # convert bc file to cpp
    COMMAND ${CMAKE_COMMAND} -DCUDA_BIN2C_EXECUTABLE:STRING="${CUDA_BIN2C_EXECUTABLE}"
                             -DCPP_FILE:STRING="${outputSource}"
                             -DCPP_SYMBOL:STRING="${exportSymbol}"
                             -DSOURCE_BASE:STRING="${outputDir}"
                             -DSOURCES:STRING="${bc_filename}"
                             -P "${bin2cpp_cmake}"
    # Remove temp bc file                         
    COMMAND ${CMAKE_COMMAND} -E remove -f "${bc_filename}"

    WORKING_DIRECTORY ${outputDir}
    MAIN_DEPENDENCY ${input}                             
    DEPENDS ${bin2cpp_cmake}
    COMMENT "Generating ${outputRelPath}/${outputName}"
    )
endfunction()

function(bc_to_cpp input outputSource outputInclude exportSymbol)
  # message("input = ${input}")
  # message("outputSource = ${outputSource}")
  # message("outputInclude = ${outputInclude}")
  # message("exportSymbol = ${exportSymbol}")
  get_filename_component(outputABS  "${outputSource}" ABSOLUTE )
  get_filename_component(outputDir  "${outputABS}" PATH )
  get_filename_component(outputName "${outputABS}" NAME )
  file(RELATIVE_PATH outputRelPath  "${EXTERNAL_BINARY_DIR}" "${outputDir}")
  get_filename_component(inputABS "${input}" ABSOLUTE)
  get_filename_component(inputDir "${inputABS}" PATH)
  get_filename_component(inputName "${inputABS}" NAME)
  
  set(bc_filename "${inputName}")
  
  # Generate header file (configure time)
  include(bin2cpp)
  bin2h(${outputInclude} ${exportSymbol} "${bc_filename}")
  
  # Convert ll to byte code
  add_custom_command(
    OUTPUT ${outputSource}

    # convert bc file to cpp
    COMMAND ${CMAKE_COMMAND} -DCUDA_BIN2C_EXECUTABLE:STRING="${CUDA_BIN2C_EXECUTABLE}"
                             -DCPP_FILE:STRING="${outputSource}"
                             -DCPP_SYMBOL:STRING="${exportSymbol}"
                             -DSOURCE_BASE:STRING="${inputDir}"
                             -DSOURCES:STRING="${bc_filename}"
                             -P "${bin2cpp_cmake}"

    WORKING_DIRECTORY ${outputDir}
    MAIN_DEPENDENCY ${input}
    DEPENDS ${bin2cpp_cmake}
    COMMENT "Generating ${outputRelPath}/${outputName}"
    )
endfunction()

################################################################################
# Compile the cpp file using clang, run an optimization pass and use bin2c to take the
# resulting code and embed it into a cpp for loading at runtime.
#
# Usage: compile_llvm_runtime( input symbol symbol output_var [clang args] )
#   input      : [in]  File to be compiled by clang
#   symbol     : [in]  Name of C symbol to use for accessing the generated code.  Also used to generate the output file names.
#   output_var : [out] Generated cpp and header files used to access compiled code at runtime
#   clang args : [in]  list of arguments to clang
#

function(compile_llvm_runtime input symbol output_var)
  set(OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}")
  get_filename_component(base "${input}" NAME_WE)
  get_filename_component(name "${input}" NAME)

  set(bc     "${OUTPUT_DIR}/${base}.bc")
  set(opt_bc "${OUTPUT_DIR}/${base}_opt.bc")
  set(opt_ll "${OUTPUT_DIR}/${base}_opt.ll")
  add_custom_command( OUTPUT ${bc} ${opt_bc} ${opt_ll}
    COMMAND ${LLVM_TOOLS_BINARY_DIR}/clang    ${input}  -o ${bc}     ${ARGN}
    COMMAND ${LLVM_TOOLS_BINARY_DIR}/opt      ${bc}     -o ${opt_bc} -always-inline -mem2reg -scalarrepl
    COMMAND ${LLVM_TOOLS_BINARY_DIR}/llvm-dis ${opt_bc} -o ${opt_ll}
    COMMENT "Compiling ${name} to ${base}_opt.ll"
    WORKING_DIRECTORY "${OUTPUT_DIR}"
    MAIN_DEPENDENCY "${input}"
    )

  set(bin2c_files
    "${CMAKE_CURRENT_BINARY_DIR}/${symbol}.cpp"
    "${CMAKE_CURRENT_BINARY_DIR}/${symbol}.h"
    )
  #bc_to_cpp(${opt_bc} ${bin2c_files} ${symbol})
  ll_to_cpp(${opt_ll} ${bin2c_files} ${symbol})

  set_source_files_properties( ${bin2c_files} PROPERTIES GENERATED TRUE )

  set(${output_var} ${bin2c_files} PARENT_SCOPE)
endfunction()


################################################################################
# Assemble the ll file to bitcode using llvm-as and use bin2c to take the
# resulting code and embed it into a cpp for loading at runtime.
#
# Usage: compile_llvm( input symbol symbol output_var [clang args] )
#   input      : [in]  File to be assembled
#   symbol     : [in]  Name of C symbol to use for accessing the generated code.  Also used to generate the output file names.
#   output_var : [out] Generated cpp and header files used to access compiled code at runtime
#

function(assemble_llvm input symbol output_var)
  set(OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}")
  get_filename_component(base "${input}" NAME_WE)
  get_filename_component(name "${input}" NAME)

  set(bin2c_files
    "${CMAKE_CURRENT_BINARY_DIR}/${symbol}.cpp"
    "${CMAKE_CURRENT_BINARY_DIR}/${symbol}.h"
    )
  #bc_to_cpp(${opt_bc} ${bin2c_files} ${symbol})
  ll_to_cpp(${input} ${bin2c_files} ${symbol})

  set_source_files_properties( ${bin2c_files} PROPERTIES GENERATED TRUE )

  set(${output_var} ${bin2c_files} PARENT_SCOPE)
endfunction()

################################################################################
# Compile the cpp file using clang
#
# Usage: cpp_to_bc( input output_var [clang args] )
#   input      : [in]  File to be compiled by clang
#   output_var : [out] Generated bc file
#   clang args : [in]  list of arguments to clang
#

function(cpp_to_bc  input output_var)
  set(OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}")
  get_filename_component(base "${input}" NAME_WE)
  get_filename_component(name "${input}" NAME)

  set(bc     "${OUTPUT_DIR}/${base}.bc")
  add_custom_command( OUTPUT ${bc}
    COMMAND ${LLVM_TOOLS_BINARY_DIR}/clang    ${input} -c -o ${bc} ${ARGN}
    COMMENT "Compiling ${name} to ${base}.bc"
    WORKING_DIRECTORY "${OUTPUT_DIR}"
    MAIN_DEPENDENCY "${input}"
    )

  set(${output_var} ${bc} PARENT_SCOPE)
endfunction()

################################################################################
# Copy ptx scripts into a string in a cpp header file.
#
# Usage: ptx_to_cpp( ptx_cpp_headers my_directory FILE1 FILE2 ... FILEN )
#   ptx_cpp_files  : [out] List of cpp files created (Note: new files are appended to this list)
#   directory      : [in]  Directory in which to place the resulting headers
#   FILE1 .. FILEN : [in]  ptx files to be cpp stringified
#
# FILE1 -> filename: ${FILE1}_ptx.cpp
#       -> string  : const char* nvrt::${FILE1}_ptx = "...";

macro( ptx_to_cpp ptx_cpp_files directory )
  foreach( file ${ARGN} )
    if( ${file} MATCHES ".*\\.ptx$" )

      #message( "file_name     : ${file}" )

      # Create the output cpp file name
      get_filename_component( base_name ${file} NAME_WE )
      set( cpp_filename ${directory}/${base_name}_ptx.cpp )
      set( variable_name ${base_name}_ptx )
      set( ptx2cpp ${CMAKE_SOURCE_DIR}/CMake/ptx2cpp.cmake )

      #message( "base_name     : ${base_name}" )
      #message( "cpp_file_name : ${cpp_filename}" )
      #message( "variable_name : ${variable_name}" )

      add_custom_command( OUTPUT ${cpp_filename}
        COMMAND ${CMAKE_COMMAND}
          -DCUDA_BIN2C_EXECUTABLE:STRING="${CUDA_BIN2C_EXECUTABLE}"
          -DCPP_FILE:STRING="${cpp_filename}"
          -DPTX_FILE:STRING="${file}"
          -DVARIABLE_NAME:STRING=${variable_name}
          -DNAMESPACE:STRING=optix
          -P ${ptx2cpp}
        DEPENDS ${file}
        DEPENDS ${ptx2cpp}
        COMMENT "${ptx2cpp}: converting ${file} to ${cpp_filename}"
        )

      list(APPEND ${ptx_cpp_files} ${cpp_filename} )
      #message( "ptx_cpp_files   : ${${ptx_cpp_files}}" )

    endif( ${file} MATCHES ".*\\.ptx$" )
  endforeach( file )
endmacro( ptx_to_cpp )


################################################################################
# Strip library of all local symbols 
#
# Usage: strip_symbols( target ) 
#   target : [out] target name for the library to be stripped 

function( strip_symbols target )
  if( NOT WIN32 )
    add_custom_command( TARGET ${target}
                        POST_BUILD
                        # using -x to strip all local symbols
                        COMMAND ${CMAKE_STRIP} -x $<TARGET_FILE:${target}>
                        COMMENT "Stripping symbols from ${target}"
                      )
  endif()
endfunction( strip_symbols )

################################################################################
# Only export the symbols that we need.
#
# Usage: optix_setup_exports( target export_file hidden_file )
#   target      : [in] target name for the library to be stripped 
#   export_file : [in] name of the file that contains the export symbol names
#   hidden_file : [in] name of the file that contains the hidden symbol names.
#                      Might be empty string in which all non-exported symbols
#                      are hidden. Only used for UNIX and NOT APPLE.
#
# Do not use this macro with WIN32 DLLs unless you are not using the dllexport
# macros. The DLL name will be set using the SOVERSION property of the target,
# so be sure to set that before calling this macro
#
function( optix_setup_exports target export_file hidden_file)
  # Suck in the exported symbol list. It should define exported_symbols.
  include(${export_file})
  # Suck in the hidden symbol list unless hidden_file is empty. It should
  # definde hidden_symbols.
  if (NOT "${hidden_file}" STREQUAL "")
   include(${hidden_file})
  endif()
  
  if( UNIX )
    if ( APPLE )
      # -exported_symbols_list lists the exact set of symbols to export.  You can call it
      # more than once if needed.
      set( export_arg -exported_symbols_list )
    else()
      # -Bsymbolic tells the linker to resolve any local symbols locally first.
      # --version-script allows us to be explicit about which symbols to export.
      set( export_arg -Bsymbolic,--version-script )
    endif()

    # Create the symbol export file.  Since Apple and Linux have different file formats
    # for doing this we will have to specify the information in the file differently.
    set(exported_symbol_file ${CMAKE_CURRENT_BINARY_DIR}/${target}_exported_symbols.txt)
    if(APPLE)
      # The base name of the symbols just has the name.  We need to prefix them with "_".
      set(modified_symbols)
      foreach(symbol ${exported_symbols})
        list(APPEND modified_symbols "_${symbol}")
      endforeach()
      # Just list the symbols.  One per line.  Since we are treating the list as a string
      # here we can replace the ';' character with a newline.
      string(REPLACE ";" "\n" exported_symbol_file_content "${modified_symbols}")
      file(WRITE ${exported_symbol_file} "${exported_symbol_file_content}\n")
    else()
      # Format is:
      #
      # {
      # global:
      # extern "C" {
      # exported_symbol;
      # };
      # local:
      # hidden_symbol; // or "*";
      # };
      # Just list the symbols.  One per line.  Since we are treating the list as a string
      # here we can insert the newline after the ';' character.
      string(REPLACE ";" ";\n" exported_symbol_file_content "${exported_symbols}")
      if (NOT "${hidden_file}" STREQUAL "")
        string(REPLACE ";" ";\n" hidden_symbol_file_content "${hidden_symbols}")
      else()
        set( hidden_symbol_file_content "*" )
      endif()
      file(WRITE ${exported_symbol_file} "{\nglobal:\nextern \"C\" {\n${exported_symbol_file_content};\n};\nlocal:\n${hidden_symbol_file_content};\n};\n")
    endif()

    # Add the command to the LINK_FLAGS
    set_property( TARGET ${target}
      APPEND_STRING 
      PROPERTY LINK_FLAGS
      " -Wl,${export_arg},${exported_symbol_file}"
      )      
  elseif( WIN32 )
    set(exported_symbol_file ${CMAKE_CURRENT_BINARY_DIR}/${target}.def)
    set(name ${target} )
    get_property( abi_version TARGET ${target} PROPERTY SOVERSION )
    if( abi_version )
      set(name "${name}.${abi_version}")
    endif()
    # Format is:
    # 
    # NAME <dllname>
    # EXPORTS
    #  <names>
    #
    string(REPLACE ";" "\n" def_file_content "${exported_symbols}" )
    file(WRITE ${exported_symbol_file} "NAME ${name}.dll\nEXPORTS\n${def_file_content}")   
    
    # Add the command to the LINK_FLAGS
    set_property( TARGET ${target}
      APPEND_STRING 
      PROPERTY LINK_FLAGS
      " /DEF:${exported_symbol_file}"
      ) 
  endif()
  
  # Make sure that if the exported_symbol_file changes we relink the library.
  set_property( TARGET ${target}
    APPEND
    PROPERTY LINK_DEPENDS
    "${exported_symbol_file}"
    )
endfunction()

################################################################################
# Some helper functions for pushing and popping variable values
#
function(push_variable variable)
  #message("push before: ${variable} = ${${variable}}, ${variable}_STACK = ${${variable}_STACK}")
  #message("             ARGN = ${ARGN}")
  #message("             ARGC = ${ARGC}, ARGV = ${ARGV}")
  if(ARGC LESS 2)
    message(FATAL_ERROR "push_variable requires at least one value to push.")
  endif()
  # Because the old value may be a list, we need to indicate how many items
  # belong to this push.  We do this by marking the start of the new push.
  list(LENGTH ${variable}_STACK start_index)
  # If the value of variable is empty, then we need to leave a placeholder,
  # because CMake doesn't have an "empty" token.
  if (DEFINED ${variable} AND NOT ${variable} STREQUAL "")
    list(APPEND ${variable}_STACK ${${variable}} ${start_index})
  else()
    list(APPEND ${variable}_STACK ${variable}_EMPTY ${start_index})
  endif()
  # Make the stack visible outside of the function's scope.
  set(${variable}_STACK ${${variable}_STACK} PARENT_SCOPE)
  # Set the new value of the variable.
  set(${variable} ${ARGN} PARENT_SCOPE)
  #set(${variable} ${ARGN}) # use for the output message below
  #message("push after : ${variable} = ${${variable}}, ${variable}_STACK = ${${variable}_STACK}")
endfunction()

function(pop_variable variable)
  #message("pop  before: ${variable} = ${${variable}}, ${variable}_STACK = ${${variable}_STACK}")
  # Find the length of the stack to use as an index to the end of the list.
  list(LENGTH ${variable}_STACK stack_length)
  if(stack_length LESS 2)
    message(FATAL_ERROR "${variable}_STACK is empty.  Can't pop any more values.")
  endif()
  math(EXPR stack_end "${stack_length} - 1")
  # Get the start of where the old value begins in the stack.
  list(GET ${variable}_STACK ${stack_end} variable_start)
  math(EXPR variable_end "${stack_end} - 1")
  foreach(index RANGE ${variable_start} ${variable_end})
    list(APPEND list_indices ${index})
  endforeach()
  list(GET ${variable}_STACK ${list_indices} stack_popped)
  # If the first element is our special EMPTY token, then we should empty it out
  if(stack_popped STREQUAL "${variable}_EMPTY")
    set(stack_popped "")
  endif()
  # Remove all the items
  list(APPEND list_indices ${stack_end})
  list(REMOVE_AT ${variable}_STACK ${list_indices})
  # Make sthe stack visible outside of the function's scope.
  set(${variable}_STACK ${${variable}_STACK} PARENT_SCOPE)
  # Assign the old value to the variable
  set(${variable} ${stack_popped} PARENT_SCOPE)
  #set(${variable} ${stack_popped}) # use for the output message below
  #message("pop  after : ${variable} = ${${variable}}, ${variable}_STACK = ${${variable}_STACK}")
endfunction()


# Helper function to generate ptx for a particular sm versions.
#
# sm_versions[input]: a list of version, such as sm_13;sm_20.  These will be used to
#                     generate the names of the output files.
# generate_files[output]: list of generated source files
# ARGN[input]: list of input CUDA C files and other options to pass to nvcc.
#
function(compile_ptx sm_versions_in generated_files)
  # CUDA_GET_SOURCES_AND_OPTIONS is a FindCUDA internal command that we are going to
  # borrow.  There are no guarantees on backward compatibility using this macro.
  CUDA_GET_SOURCES_AND_OPTIONS(_sources _cmake_options _options ${ARGN})
  # Check to see if they specified an sm version, and spit out an error.
  list(FIND _options -arch arch_index)
  if(arch_index GREATER -1)
    math(EXPR sm_index "${arch_index}+1")
    list(GET _options ${sm_index} sm_value)
    message(FATAL_ERROR "-arch ${sm_value} has been specified to compile_ptx.  Please remove that option and put it in the sm_versions argument.")
  endif()
  set(${generated_files})
  set(sm_versions ${sm_versions_in})
  if(NOT CUDA_SM_20)
    list(REMOVE_ITEM sm_versions sm_20)
  endif()
  push_variable(CUDA_64_BIT_DEVICE_CODE ON)
  foreach(source ${_sources})
    set(ptx_generated_files)
    #message("\n\nProcessing ${source}")
    foreach(sm ${sm_versions})

      # Generate the 32 bit ptx for 32 bit builds and when the CMAKE_OSX_ARCHITECTURES
      # specifies it.
      list(FIND CMAKE_OSX_ARCHITECTURES i386 osx_build_32_bit_ptx)
      if( CMAKE_SIZEOF_VOID_P EQUAL 4 OR NOT osx_build_32_bit_ptx LESS 0)
        set(CUDA_64_BIT_DEVICE_CODE OFF)
        CUDA_WRAP_SRCS( ptx_${sm}_32 PTX _generated_files ${source} ${_cmake_options}
          OPTIONS -arch ${sm} ${_options}
          )
        # Add these files onto the list of files.
        list(APPEND ptx_generated_files ${_generated_files})
      endif()

      # Generate the 64 bit ptx for 64 bit builds and when the CMAKE_OSX_ARCHITECTURES
      # specifies it.
      list(FIND CMAKE_OSX_ARCHITECTURES x86_64 osx_build_64_bit_ptx)
      if( CMAKE_SIZEOF_VOID_P EQUAL 8 OR NOT osx_build_64_bit_ptx LESS 0)
        set(CUDA_64_BIT_DEVICE_CODE ON)
        CUDA_WRAP_SRCS( ptx_${sm}_64 PTX _generated_files ${source} ${_cmake_options}
          OPTIONS -arch ${sm} ${_options}
          )
        # Add these files onto the list of files.
        list(APPEND ptx_generated_files ${_generated_files})
      endif()
    endforeach()

    get_filename_component(source_basename "${source}" NAME_WE)
    set(cpp_wrap "${CMAKE_CURRENT_BINARY_DIR}/${source_basename}_ptx.cpp")
    set(h_wrap   "${CMAKE_CURRENT_BINARY_DIR}/${source_basename}_ptx.h")

    set(relative_ptx_generated_files)
    foreach(file ${ptx_generated_files})
      get_filename_component(fname "${file}" NAME)
      list(APPEND relative_ptx_generated_files "${fname}")
    endforeach()

    # Now generate a target that will generate the wrapped version of the ptx
    # files at build time
    set(symbol "${source_basename}_source")
    add_custom_command( OUTPUT ${cpp_wrap}
      COMMAND ${CMAKE_COMMAND} -DCUDA_BIN2C_EXECUTABLE:STRING="${CUDA_BIN2C_EXECUTABLE}"
      -DCPP_FILE:STRING="${cpp_wrap}"
      -DCPP_SYMBOL:STRING="${symbol}"
      -DSOURCE_BASE:STRING="${CMAKE_CURRENT_BINARY_DIR}"
      -DSOURCES:STRING="${relative_ptx_generated_files}"
      ARGS -P "${CMAKE_SOURCE_DIR}/CMake/bin2cpp.cmake"
      DEPENDS ${CMAKE_SOURCE_DIR}/CMake/bin2cpp.cmake ${ptx_generated_files}
      )
    # We know the list of files at configure time, so generate the files here
    include(bin2cpp)
    bin2h("${h_wrap}" ${symbol} ${relative_ptx_generated_files})

    list(APPEND ${generated_files} ${ptx_generated_files} ${cpp_wrap} ${h_wrap})
  endforeach(source)
  pop_variable(CUDA_64_BIT_DEVICE_CODE)

  set(${generated_files} ${${generated_files}} PARENT_SCOPE)
endfunction()

# Helper function to generate the appropiate options for a CUDA compile
# based on the target architectures.
#
# Function selects the higher compute capability available and generates code for that one.
#
# Usage: cuda_generate_runtime_target_options( output_var target_list )
# 
# output[output] is a list-variable to fill with options for CUDA_COMPILE
# ARGN[input] is a list of targets, i.e. sm_11 sm_20 sm_30
#             NO_PTX in the input list will not add PTX to the highest SM version

function( cuda_generate_runtime_target_options output )
  # remove anything that is not sm_XX, and look for NO_PTX option
  set( no_ptx FALSE )
  foreach(target ${ARGN})
    string( REGEX MATCH "^(sm_[0-9][0-9])$" match ${target} )
    if( NOT CMAKE_MATCH_1 )
      list( REMOVE_ITEM ARGN ${target} )
    endif( NOT CMAKE_MATCH_1 )
    if( target STREQUAL "NO_PTX" )
      set( no_ptx TRUE )
    endif()
  endforeach(target)

  list( LENGTH ARGN valid_target_count )
  if( valid_target_count GREATER 0 )

# We will add compute_XX automatically, infer max compatible compute capability.
    # check targets for max compute capability
    set( smver_max "0" )
    foreach(target ${ARGN})
      string( REGEX MATCH "sm_([0-9][0-9])$" sm_ver_match ${target} )
      if( CMAKE_MATCH_1 )
        if( ${CMAKE_MATCH_1} STRGREATER smver_max )
          set( smver_max ${CMAKE_MATCH_1} )
        endif( ${CMAKE_MATCH_1} STRGREATER smver_max )
      endif( CMAKE_MATCH_1 )
      unset( sm_ver_match )
    endforeach(target)

    if( no_ptx )
      set( smver_max "You can't match me, I'm the ginger bread man!" )
    endif()
    
    # copy the input list to a new one and sort it
    set( sm_versions ${ARGN} )
    list( SORT sm_versions )
    
    # walk to SM versions to generate the entries of gencode
    set( options "" )
    foreach( sm_ver ${sm_versions} )
      string( REGEX MATCH "sm_([0-9][0-9])$" sm_ver_num ${sm_ver} )  

# This adds compute_XX automatically, in order to generate PTX.
      if( ${CMAKE_MATCH_1} STREQUAL ${smver_max} )
        # append the max compute capability, to get compute_XX too.
        # this appends the PTX code for the higher SM_ version
        set(entry -gencode=arch=compute_${CMAKE_MATCH_1},code=\\\"${sm_ver},compute_${smver_max}\\\")
      else( ${CMAKE_MATCH_1} STREQUAL ${smver_max} )
        set(entry -gencode=arch=compute_${CMAKE_MATCH_1},code=\\\"${sm_ver}\\\")     
      endif( ${CMAKE_MATCH_1} STREQUAL ${smver_max} )
      list( APPEND options ${entry} )
    endforeach( sm_ver ${sm_versions} )
    
    # return the generated option string
    set( ${output} ${options} PARENT_SCOPE )
    
    unset( smver_max )
    unset( sm_versions )
  else( valid_target_count GREATER 0 )
    # return empty string
    set( ${output} "" PARENT_SCOPE )    
  endif( valid_target_count GREATER 0 )

endfunction(cuda_generate_runtime_target_options)


# Compile the list of SASS assembler files to cubins. 
# Then take the resulting file and store it in a cpp file using bin2c. 
#
# Usage: compile_sass_to_cpp( _generated_files files [files...] [OPTIONS ...] )
#
# _generated_files[output] is a list-variable to fill with the names of the generated files
# ARGN[input] is a list of files and nvasm_internal options.  

function(compile_sass_to_cpp _generated_files )
  # CUDA_GET_SOURCES_AND_OPTIONS is a FindCUDA internal command that we are going to
  # borrow.  There are no guarantees on backward compatibility using this macro.
  CUDA_GET_SOURCES_AND_OPTIONS(_sources _cmake_options _options ${ARGN})

  set(generated_files)
  foreach(source ${_sources})

    get_filename_component(source_basename ${source} NAME_WE)
    set(cubinfile ${CMAKE_CURRENT_BINARY_DIR}/${source_basename}.cubin)

    set(source ${CMAKE_CURRENT_SOURCE_DIR}/${source})
 
    set(cuda_build_comment_string "Assembling to cubin file ${source}" )
    set_source_files_properties(${source} PROPERTIES HEADER_FILE_ONLY TRUE)
 
    add_custom_command(OUTPUT ${cubinfile} 
      COMMAND ${CUDA_NVASM_EXECUTABLE} ${_options} ${source} -o ${cubinfile}
      DEPENDS ${source}
      COMMENT "${cuda_build_comment_string}"
      )

    set(cpp_wrap "${CMAKE_CURRENT_BINARY_DIR}/${source_basename}_cuda.cpp")
    set(h_wrap   "${CMAKE_CURRENT_BINARY_DIR}/${source_basename}_cuda.h")

    get_filename_component(generated_file_path "${cubinfile}" DIRECTORY)
    get_filename_component(relative_cuda_generated_file "${cubinfile}" NAME)
 
    # Now generate a target that will generate the wrapped version of the cuda
    # files at build time
    set(symbol "${source_basename}_cuda_source")
    add_custom_command( OUTPUT ${cpp_wrap}
       COMMAND ${CMAKE_COMMAND} -DCUDA_BIN2C_EXECUTABLE:STRING="${CUDA_BIN2C_EXECUTABLE}"
       -DCPP_FILE:STRING="${cpp_wrap}"
       -DCPP_SYMBOL:STRING="${symbol}"
       -DSOURCE_BASE:STRING="${generated_file_path}"
       -DSOURCES:STRING="${relative_cuda_generated_file}"
       ARGS -P "${CMAKE_SOURCE_DIR}/CMake/bin2cpp.cmake"
       DEPENDS ${CMAKE_SOURCE_DIR}/CMake/bin2cpp.cmake ${cubinfile}
       )
     # We know the list of files at configure time, so generate the files here
     include(bin2cpp)
     bin2h("${h_wrap}" ${symbol} ${relative_cuda_generated_files})
  
     list(APPEND generated_files ${cubinfile} ${cpp_wrap} ${h_wrap})

  endforeach()
  
  set_source_files_properties(${generated_files} PROPERTIES GENERATED TRUE)
  set(${_generated_files} ${generated_files} PARENT_SCOPE)
endfunction()


# Compile the list of cuda files using the specified format.  Then take the resulting file
# and store it in a cpp file using bin2c.  This is not appropriate for PTX formats.  Use
# compile_ptx for that.
#
# Usage: compile_cuda_to_cpp( target_name format _generated_files files [files...] [OPTIONS ...] )
#
# target_name[input] name to use for mangling output files.
# format[input] OBJ SEPARABLE_OBJ CUBIN FATBIN
# _generated_files[output] is a list-variable to fill with the names of the generated files
# ARGN[input] is a list of files and optional CUDA_WRAP_SRCS options.  See documentation
# for CUDA_WRAP_SRCS in FindCUDA.cmake.

function(compile_cuda_to_cpp target_name format _generated_files)
  # CUDA_GET_SOURCES_AND_OPTIONS is a FindCUDA internal command that we are going to
  # borrow.  There are no guarantees on backward compatibility using this macro.
  CUDA_GET_SOURCES_AND_OPTIONS(_sources _cmake_options _options ${ARGN})

  if(${format} MATCHES "SEPARABLE_OBJ")
    # It's OK to set this without resetting it later, since this is a function with a
    # localized scope.
    set(CUDA_SEPARABLE_COMPILATION ON)
    set(format OBJ)
  elseif(${format} MATCHES "PTX")
    message(FATAL_ERROR "compile_cuda_to_cpp called with PTX format which is unsupported.  Try compile_ptx instead.")
  endif()

  set(${_generated_files})
  foreach(source ${_sources})
    CUDA_WRAP_SRCS(${target_name} ${format} objfile ${source} OPTIONS ${_options} )

    get_filename_component(source_basename "${source}" NAME_WE)
    set(cpp_wrap "${CMAKE_CURRENT_BINARY_DIR}/${source_basename}_cuda.cpp")
    set(h_wrap   "${CMAKE_CURRENT_BINARY_DIR}/${source_basename}_cuda.h")

    get_filename_component(generated_file_path "${objfile}" DIRECTORY)
    get_filename_component(relative_cuda_generated_file "${objfile}" NAME)

    # Now generate a target that will generate the wrapped version of the cuda
    # files at build time
    set(symbol "${source_basename}_cuda_source")
    add_custom_command( OUTPUT ${cpp_wrap}
      COMMAND ${CMAKE_COMMAND} -DCUDA_BIN2C_EXECUTABLE:STRING="${CUDA_BIN2C_EXECUTABLE}"
      -DCPP_FILE:STRING="${cpp_wrap}"
      -DCPP_SYMBOL:STRING="${symbol}"
      -DSOURCE_BASE:STRING="${generated_file_path}"
      -DSOURCES:STRING="${relative_cuda_generated_file}"
      ARGS -P "${CMAKE_SOURCE_DIR}/CMake/bin2cpp.cmake"
      DEPENDS ${CMAKE_SOURCE_DIR}/CMake/bin2cpp.cmake ${objfile}
      )
    # We know the list of files at configure time, so generate the files here
    include(bin2cpp)
    bin2h("${h_wrap}" ${symbol} ${relative_cuda_generated_files})

    list(APPEND ${_generated_files} ${objfile} ${cpp_wrap} ${h_wrap})
  endforeach()
  set(${_generated_files} ${${_generated_files}} PARENT_SCOPE)
endfunction()

# Compile the list of cuda files using the specified format.  Then take the resulting file
# and store it in a cpp file using bin2c. Appends the variant name to the source file name.
#
# Usage: compile_cuda_to_cpp_variant( target_name variant_name format _generated_files files [files...] [OPTIONS ...] )
#
# target_name[input] name to use for mangling output files.
# variant_name[input] name to append to filenames.
# format[input] OBJ SEPARABLE_OBJ CUBIN FATBIN
# _generated_files[output] is a list-variable to fill with the names of the generated files
# ARGN[input] is a list of files and optional CUDA_WRAP_SRCS options.  See documentation
# for CUDA_WRAP_SRCS in FindCUDA.cmake.

function(compile_cuda_to_cpp_variant target_name variant_name format _generated_files)
  # CUDA_GET_SOURCES_AND_OPTIONS is a FindCUDA internal command that we are going to
  # borrow.  There are no guarantees on backward compatibility using this macro.
  CUDA_GET_SOURCES_AND_OPTIONS(_sources _cmake_options _options ${ARGN})

  if(${format} MATCHES "SEPARABLE_OBJ")
    # It's OK to set this without resetting it later, since this is a function with a
    # localized scope.
    set(CUDA_SEPARABLE_COMPILATION ON)
    set(format OBJ)
  elseif(${format} MATCHES "PTX")
    message(FATAL_ERROR "compile_cuda_to_cpp called with PTX format which is unsupported.  Try compile_ptx instead.")
  endif()

  set(${_generated_files})
  foreach(source ${_sources})
    get_filename_component(source_basename "${source}" NAME_WE)

    CUDA_WRAP_SRCS(${target_name}_${variant_name} ${format} objfile ${source} OPTIONS ${_options} )

    set(cpp_wrap "${CMAKE_CURRENT_BINARY_DIR}/${source_basename}_${variant_name}_cuda.cpp")
    set(h_wrap   "${CMAKE_CURRENT_BINARY_DIR}/${source_basename}_${variant_name}_cuda.h")

    get_filename_component(generated_file_path "${objfile}" DIRECTORY)
    get_filename_component(relative_cuda_generated_file "${objfile}" NAME)

    # Now generate a target that will generate the wrapped version of the cuda
    # files at build time
    set(symbol "${source_basename}_${variant_name}_cuda_source")
    add_custom_command( OUTPUT ${cpp_wrap}
      COMMAND ${CMAKE_COMMAND} -DCUDA_BIN2C_EXECUTABLE:STRING="${CUDA_BIN2C_EXECUTABLE}"
      -DCPP_FILE:STRING="${cpp_wrap}"
      -DCPP_SYMBOL:STRING="${symbol}"
      -DSOURCE_BASE:STRING="${generated_file_path}"
      -DSOURCES:STRING="${relative_cuda_generated_file}"
      ARGS -P "${CMAKE_SOURCE_DIR}/CMake/bin2cpp.cmake"
      DEPENDS ${CMAKE_SOURCE_DIR}/CMake/bin2cpp.cmake ${objfile}
      )
    # We know the list of files at configure time, so generate the files here
    include(bin2cpp)
    bin2h("${h_wrap}" ${symbol} ${relative_cuda_generated_files})

    list(APPEND ${_generated_files} ${objfile} ${cpp_wrap} ${h_wrap})
  endforeach()
  set(${_generated_files} ${${_generated_files}} PARENT_SCOPE)
endfunction()

# Compile the list of ptx files.  Then take the resulting file
# and store it in a cpp file using bin2c.
#

function(compile_ptx_to_cpp input_ptx _generated_files extra_dependencies )
  get_filename_component(source_ptx "${input_ptx}" REALPATH )
  get_filename_component(source_basename "${source_ptx}" NAME_WE)
  message("source_ptx ${source_ptx}")
  message("source_basename ${source_basename}")
  set(fatbin ${CMAKE_CURRENT_BINARY_DIR}/${source_basename}.fatbin)
  set(build_directory "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles")
  
  # the source is passed in as the first parameter
  # use this function to easily extract options
  CUDA_GET_SOURCES_AND_OPTIONS( _ptx_wrap_sources _ptx_wrap_cmake_options _ptx_wrap_options ${ARGN})

  # extract all macro definitions so we can pass these to the c preprocessor
  string(REGEX MATCHALL "-D[A-z_][A-z_0-9]*(=[A-z_0-9]*)?" macros ${ARGN})

  set(fatbin_command)
  set(cubin_commands)

  if (MSVC)
    set(preprocess_command CL /nologo /E /EP)
  elseif(CMAKE_COMPILER_IS_GNUCC)
    set(preprocess_command gcc -P -E -x assembler-with-cpp )
  else()
    message(FATAL_ERROR "Unknown preprocessor command")
  endif()

  foreach(cuda_sm_target ${cuda_sm_targets})
    string(REGEX REPLACE "sm_" "" cuda_sm "${cuda_sm_target}")

    # Take input PTX and generate cubin
    set(preprocessed_ptx ${build_directory}/${source_basename}.${cuda_sm}.ptx)
    set(cubin            ${build_directory}/${source_basename}.${cuda_sm}.cubin)

    list(APPEND cubin_commands
      COMMAND ${preprocess_command} -DPTXAS  ${macros} ${PTXAS_INCLUDES} -D__CUDA_TARGET__=${cuda_sm} -D__CUDA_ARCH__=${cuda_sm}0 ${source_ptx} > ${preprocessed_ptx}
      COMMAND ${CUDA_NVCC_EXECUTABLE} ${_ptx_wrap_options} -arch=sm_${cuda_sm} --cubin ${preprocessed_ptx} -dc -o ${cubin}
      )
    list(APPEND fatbin_command "--image=profile=sm_${cuda_sm},file=${cubin}")
  endforeach()

  add_custom_command(
    OUTPUT ${fatbin}
    ${cubin_commands}
    COMMAND ${CUDA_FATBINARY_EXECUTABLE} --create="${fatbin}" -64 -c --cmdline="" ${fatbin_command}
    MAIN_DEPENDENCY ${source_ptx}
    DEPENDS ${extra_dependencies}
    )

  set(cpp_wrap "${CMAKE_CURRENT_BINARY_DIR}/${source_basename}.cpp")
  set(h_wrap   "${CMAKE_CURRENT_BINARY_DIR}/${source_basename}.h")
  get_filename_component(generated_file_path "${fatbin}" DIRECTORY)
  get_filename_component(relative_generated_file "${fatbin}" NAME)
  
  set(symbol "${source_basename}")
  add_custom_command( OUTPUT ${cpp_wrap}
    COMMAND ${CMAKE_COMMAND} -DCUDA_BIN2C_EXECUTABLE:STRING="${CUDA_BIN2C_EXECUTABLE}"
    -DCPP_FILE:STRING="${cpp_wrap}"
    -DCPP_SYMBOL:STRING="${symbol}"
    -DSOURCE_BASE:STRING="${generated_file_path}"
    -DSOURCES:STRING="${relative_generated_file}"
    ARGS -P "${bin2cpp_cmake}"
    DEPENDS ${CMAKE_SOURCE_DIR}/CMake/bin2cpp.cmake ${fatbin}
    )

  bin2h("${h_wrap}" "${symbol}" "${relative_generated_file}")

  set(${_generated_files} ${cpp_wrap} ${h_wrap} PARENT_SCOPE)
endfunction()

# Create multiple bitness targets for mac universal builds
#
# Usage: OPTIX_MAKE_UNIVERSAL_CUDA_RUNTIME_OBJECTS( 
#          target_name
#          generated_files_var
#          FILE0.cu FILE1.cu ... FILEN.cu
#          OPTIONS
#          option1 option2
#          )
# 
# target_name     [input ] name prefix for the resulting files 
# generated_files [output] list of filenames of resulting object files
# ARGN            [input ] is a list of source files plus possibly options 
function( OPTIX_MAKE_UNIVERSAL_CUDA_RUNTIME_OBJECTS target_name generated_files ) 

  # If you specified CMAKE_OSX_ARCHITECTURES that means you want a universal build (though
  # this should work for a single architecture).
  if (CMAKE_OSX_ARCHITECTURES)
    set(cuda_files)

    push_variable(CUDA_64_BIT_DEVICE_CODE OFF)
    list(LENGTH CMAKE_OSX_ARCHITECTURES num_arches)
    if (num_arches GREATER 1)
      # If you have more than one architecture then you don't want to attach the build rule
      # to the file itself otherwise you could compile some files in multiple targets.
      set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)
    endif()
    foreach(arch ${CMAKE_OSX_ARCHITECTURES})
      # Set the bitness of the build specified by nvcc to the one matching the OSX
      # architecture.
      if (arch STREQUAL "i386")
        set(CUDA_64_BIT_DEVICE_CODE OFF)
      elseif (arch STREQUAL "x86_64")
        set(CUDA_64_BIT_DEVICE_CODE ON)
      else()
        message(SEND_ERROR "Unknown OSX arch ${arch}")
      endif()

      CUDA_WRAP_SRCS(
        ${target_name}_${arch}
        OBJ
        cuda_sources
        ${ARGN}
        )
      list( APPEND cuda_files ${cuda_sources} )
    endforeach()
    pop_variable(CUDA_64_BIT_DEVICE_CODE)

  else()
    CUDA_WRAP_SRCS(
      ${target_name}
      OBJ
      cuda_files
      ${ARGN}
      )
  endif()

  set( ${generated_files} ${cuda_files} PARENT_SCOPE )
endfunction()

################################################################
# Simple debugging function for printing the value of a variable
#
# USAGE: print_var(
#   var
#   msg (optional)
#  )
function(print_var var)
  set(msg "")
  if (ARGC GREATER 1)
    set(msg "${ARGV1}\n  ")
  endif()
  message("${msg}${var}:${${var}}")
endfunction()

################################################################
# Function for adding a list of files from a subdirectory.  Also adds an IDE source group.
#
# dir             - name of the sub directory
# source_list_var - name of the variable to set the list of sources to
# ARGN            - list of sources relative to the sub directory
function(optix_add_subdirectory_sources dir source_list_var)
  set(files ${ARGN})
  set(sources)
  string( REPLACE "${CMAKE_CURRENT_SOURCE_DIR}/" "" rel_dir ${dir} ) 

  foreach(filename ${files})
    if(NOT IS_ABSOLUTE ${filename})
      set(filename ${rel_dir}/${filename})
    endif()
    list(APPEND sources ${filename})
  endforeach()

  # Make a source group
  string( REPLACE "/" "\\\\" group_name ${rel_dir} )
  if( group_name )
    source_group( ${group_name} FILES ${sources} )
  endif()   

  set(${source_list_var} ${sources} PARENT_SCOPE)
endfunction()

################################################################
# optixSharedLibraryResources
#
# Handle common logic for Windows resource script generation for shared
# libraries.  The variable 'resourceFiles' is set on the parent scope for
# use in add_library.
#
# outputName - The value to assign to 'output_name' in the parent scope
#              and used to locate the resource script template to configure.
#
# Side effects:
#
# output_name   - Value to be used for the OUTPUT_NAME property of the target.
# resourceFiles - List of resource files to be added to the target.
#
macro( optixSharedLibraryResources outputName )
  set( resourceFiles )
  set( output_name ${outputName} )
  if( WIN32 )
    # On Windows, we want the version number in the DLL filename.
    # Windows ignores the SOVERSION property and only uses the OUTPUT_NAME property.
    # Linux puts the version number in the filename from the SOVERSION property.
    # We need to adjust this variable before we call configure_file.
    set( output_name "${outputName}.${OPTIX_SO_VERSION}" )
    configure_file( "${outputName}.rc.in" "${outputName}.rc" @ONLY )
    set( resourceFiles "${CMAKE_CURRENT_BINARY_DIR}/${outputName}.rc" "${CMAKE_BINARY_DIR}/include/optix_rc.h" )
    source_group( Resources FILES ${resourceFiles} )
  endif()
endmacro()

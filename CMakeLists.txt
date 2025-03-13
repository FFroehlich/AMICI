#
# Build AMICI library
#
cmake_minimum_required(VERSION 3.22)
# When updating the policy version, please also update it in
# src/CMakeLists.template.cmake
cmake_policy(VERSION 3.22...3.31)

project(amici)

# misc options
if(DEFINED ENV{AMICI_TRY_ENABLE_HDF5})
  option(AMICI_TRY_ENABLE_HDF5 "Build with HDF5 support if available?"
         $ENV{AMICI_TRY_ENABLE_HDF5})
else()
  option(AMICI_TRY_ENABLE_HDF5 "Build with HDF5 support if available?" ON)
endif()
option(AMICI_PYTHON_BUILD_EXT_ONLY "Build only the Python extension?" OFF)
option(ENABLE_HDF5 "Build with HDF5 support?" OFF)
option(SUNDIALS_SUPERLUMT_ENABLE "Enable sundials SuperLUMT?" OFF)
option(EXPORT_PACKAGE "Export AMICI library to CMake package registry?" ON)
option(ENABLE_SWIG "Build AMICI swig library?" ON)
option(ENABLE_PYTHON "Create Python module?" ON)
option(BUILD_TESTS "Build integration tests?" ON)

message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR}")
message(STATUS "CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}"
                                                STREQUAL "MINGW")
  # require at least gcc 9.1 for proper C+17 support, e.g. std::reduce
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.1)
    message(FATAL_ERROR "GCC version must be at least 9.1!")
  endif()
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags
include(CheckCXXCompilerFlag)
set(MY_CXX_FLAGS -Wall)
foreach(flag ${MY_CXX_FLAGS})
  unset(CUR_FLAG_SUPPORTED CACHE)
  check_cxx_compiler_flag(${flag} CUR_FLAG_SUPPORTED)
  if(${CUR_FLAG_SUPPORTED})
    string(APPEND CMAKE_CXX_FLAGS " ${flag}")
  endif()
endforeach()

if(MSVC)
  set(MY_CXX_FLAGS
      # hide copyright message
      -nologo
      # -wd4820: hide padding
      -wd4820
      # hide unreferenced inline function removed
      -wd4514
      # hide not inlined
      -wd4710
      # hide auto-inlined
      -wd4711
      # hide copy-ctor implicitly deleted
      -wd4625
      # hide assignment-ctor implicitly deleted
      -wd4626
      # hide spectre mitigation
      -wd5045
      # hide enum-switch no explicitly handled
      -wd4061)
  foreach(flag ${MY_CXX_FLAGS})
    check_cxx_compiler_flag(${flag} CUR_FLAG_SUPPORTED)
    if(${CUR_FLAG_SUPPORTED})
      string(APPEND CMAKE_CXX_FLAGS " ${flag}")
    endif()
  endforeach()
endif()

# Debug build?
if("$ENV{ENABLE_AMICI_DEBUGGING}" OR "$ENV{ENABLE_GCOV_COVERAGE}")
  add_compile_options(-UNDEBUG)
  if(MSVC)
    add_compile_options(-DEBUG)
  else()
    add_compile_options(-O0 -g)
  endif()
endif()

# coverage options
if($ENV{ENABLE_GCOV_COVERAGE})
  string(APPEND CMAKE_CXX_FLAGS_DEBUG " --coverage")
  string(APPEND CMAKE_EXE_LINKER_FLAGS_DEBUG " --coverage")
endif()

# Legacy environment variables from Python-sdist times
if(DEFINED ENV{AMICI_CXXFLAGS})
  message(STATUS "Appending flags from AMICI_CXXFLAGS: $ENV{AMICI_CXXFLAGS}")
  add_compile_options("$ENV{AMICI_CXXFLAGS}")
endif()
if(DEFINED ENV{AMICI_LDFLAGS})
  message(STATUS "Appending flags from AMICI_LDFLAGS: $ENV{AMICI_LDFLAGS}")
  link_libraries("$ENV{AMICI_LDFLAGS}")
endif()

if(DEFINED ENV{SWIG})
  message(STATUS "Setting SWIG_EXECUTABLE to $ENV{SWIG} ($SWIG)")
  unset(SWIG_VERSION CACHE)
  unset(SWIG_DIR CACHE)
  set(SWIG_EXECUTABLE $ENV{SWIG})
endif()

# find dependencies
include(GNUInstallDirs)

include(AmiciFindBLAS)
find_package(OpenMP)
find_package(Boost COMPONENTS chrono)

if(ENABLE_HDF5)
  find_package(
    HDF5
    COMPONENTS C HL CXX
    REQUIRED)
elseif(AMICI_TRY_ENABLE_HDF5)
  find_package(HDF5 COMPONENTS C HL CXX)
endif()

set(VENDORED_SUNDIALS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/sundials)
set(SUNDIALS_PRIVATE_INCLUDE_DIRS
    "${VENDORED_SUNDIALS_DIR}/src;${VENDORED_SUNDIALS_DIR}/src/sundials")
# Handle different sundials build/install dirs, depending on whether we are
# building the Python extension only or the full C++ interface
if(AMICI_PYTHON_BUILD_EXT_ONLY)
  set(VENDORED_SUNDIALS_BUILD_DIR ${CMAKE_CURRENT_SOURCE_DIR})
  set(VENDORED_SUNDIALS_INSTALL_DIR ${VENDORED_SUNDIALS_BUILD_DIR})
else()
  set(VENDORED_SUNDIALS_BUILD_DIR ${VENDORED_SUNDIALS_DIR}/build)
  set(VENDORED_SUNDIALS_INSTALL_DIR ${VENDORED_SUNDIALS_BUILD_DIR})
endif()
set(SUNDIALS_ROOT "${VENDORED_SUNDIALS_INSTALL_DIR}/${CMAKE_INSTALL_LIBDIR}")
find_package(SUNDIALS REQUIRED CONFIG PATHS "${SUNDIALS_ROOT}/cmake/sundials/")
message(STATUS "Found SUNDIALS: ${SUNDIALS_DIR}")

set(GSL_LITE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/gsl")

# Add target to create version file
add_custom_target(
  version
  ${CMAKE_COMMAND}
  -D
  SRC=${CMAKE_CURRENT_SOURCE_DIR}/include/amici/version.in.h
  -D
  DST=${CMAKE_CURRENT_BINARY_DIR}/include/amici/version.h
  -P
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/configureVersion.cmake
  COMMENT "Writing amici/version.h")

# Library source files
set(AMICI_SRC_LIST
    src/symbolic_functions.cpp
    src/splinefunctions.cpp
    src/cblas.cpp
    src/amici.cpp
    src/misc.cpp
    src/rdata.cpp
    src/edata.cpp
    src/exception.cpp
    src/simulation_parameters.cpp
    src/spline.cpp
    src/solver.cpp
    src/solver_cvodes.cpp
    src/solver_idas.cpp
    src/logging.cpp
    src/model.cpp
    src/model_ode.cpp
    src/model_dae.cpp
    src/model_state.cpp
    src/newton_solver.cpp
    src/forwardproblem.cpp
    src/steadystateproblem.cpp
    src/backwardproblem.cpp
    src/sundials_matrix_wrapper.cpp
    src/sundials_linsol_wrapper.cpp
    src/abstract_model.cpp
    src/vector.cpp
    include/amici/abstract_model.h
    include/amici/amici.h
    include/amici/backwardproblem.h
    include/amici/cblas.h
    include/amici/defines.h
    include/amici/edata.h
    include/amici/exception.h
    include/amici/forwardproblem.h
    include/amici/hdf5.h
    include/amici/logging.h
    include/amici/misc.h
    include/amici/model_dae.h
    include/amici/model_dimensions.h
    include/amici/model.h
    include/amici/model_ode.h
    include/amici/model_state.h
    include/amici/newton_solver.h
    include/amici/rdata.h
    include/amici/serialization.h
    include/amici/simulation_parameters.h
    include/amici/solver_cvodes.h
    include/amici/solver.h
    include/amici/solver_idas.h
    include/amici/spline.h
    include/amici/splinefunctions.h
    include/amici/steadystateproblem.h
    include/amici/sundials_linsol_wrapper.h
    include/amici/sundials_matrix_wrapper.h
    include/amici/symbolic_functions.h
    include/amici/vector.h
    $<$<BOOL:${HDF5_FOUND}>:src/hdf5.cpp>)

add_library(${PROJECT_NAME} ${AMICI_SRC_LIST})

set(AMICI_CXX_OPTIONS
    ""
    CACHE STRING "C++ options for libamici (semicolon-separated)")
target_compile_options(${PROJECT_NAME} PRIVATE "${AMICI_CXX_OPTIONS}")
set_property(
  SOURCE src/solver_cvodes.cpp
  APPEND
  PROPERTY INCLUDE_DIRECTORIES "${SUNDIALS_PRIVATE_INCLUDE_DIRS}")
set_property(
  SOURCE src/solver_idas.cpp
  APPEND
  PROPERTY INCLUDE_DIRECTORIES "${SUNDIALS_PRIVATE_INCLUDE_DIRS}")
add_dependencies(${PROJECT_NAME} version)

file(GLOB PUBLIC_HEADERS include/amici/*.h)
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER
                                                 "${PUBLIC_HEADERS}")
target_include_directories(
  ${PROJECT_NAME}
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
         $<BUILD_INTERFACE:${GSL_LITE_INCLUDE_DIR}>
         $<INSTALL_INTERFACE:${CMAKE_INSTALL_DATADIR}/amici/swig>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/swig>)

if("$ENV{ENABLE_AMICI_DEBUGGING}"
   OR "$ENV{ENABLE_GCOV_COVERAGE}"
   OR CMAKE_BUILD_TYPE MATCHES "Debug")
  set(MY_CXX_FLAGS -Werror -Wno-error=deprecated-declarations)
  foreach(flag ${MY_CXX_FLAGS})
    unset(CUR_FLAG_SUPPORTED CACHE)
    check_cxx_compiler_flag(${flag} CUR_FLAG_SUPPORTED)
    if(${CUR_FLAG_SUPPORTED})
      target_compile_options(${PROJECT_NAME} PRIVATE ${flag})
    endif()
  endforeach()
endif()

target_compile_definitions(
  ${PROJECT_NAME} PUBLIC $<$<BOOL:${Boost_CHRONO_FOUND}>:HAS_BOOST_CHRONO>)

target_link_libraries(
  ${PROJECT_NAME}
  PUBLIC SUNDIALS::nvecserial_static
         SUNDIALS::sunmatrixband_static
         SUNDIALS::sunmatrixdense_static
         SUNDIALS::sunmatrixsparse_static
         SUNDIALS::sunlinsolspbcgs_static
         SUNDIALS::sunlinsolspfgmr_static
         SUNDIALS::sunlinsolspgmr_static
         SUNDIALS::sunlinsolsptfqmr_static
         SUNDIALS::cvodes_static
         SUNDIALS::idas_static
         $<$<BOOL:${Boost_CHRONO_FOUND}>:Boost::chrono>
         $<$<BOOL:${OpenMP_FOUND}>:OpenMP::OpenMP_CXX>
         ${CMAKE_DL_LIBS}
  PRIVATE
    BLAS::BLAS
    SUNDIALS::sunnonlinsolnewton_static
    SUNDIALS::sunnonlinsolfixedpoint_static
    SUNDIALS::sunlinsolklu_static
    SUNDIALS::sunlinsolband_static
    SUNDIALS::sunlinsoldense_static
    SUNDIALS::sunlinsolpcg_static
    $<$<BOOL:${SUNDIALS_SUPERLUMT_ENABLE}>:SUNDIALS::sundials_sunlinsolsuperlumt>
)

if(HDF5_FOUND)
  message(STATUS "HDF5 library found. Building AMICI with HDF5 support.")
  target_link_libraries(${PROJECT_NAME} PUBLIC hdf5::hdf5_hl_cpp hdf5::hdf5_hl
                                               hdf5::hdf5_cpp hdf5::hdf5)
else()
  message(
    STATUS
      "HDF5 support disabled or HDF5 library not found. Building AMICI WITHOUT HDF5 support."
  )
endif()

if(AMICI_PYTHON_BUILD_EXT_ONLY)
  add_compile_definitions("gsl_CONFIG_CONTRACT_VIOLATION_THROWS"
                          "gsl_CONFIG_NARROW_THROWS_ON_TRUNCATION=1")
endif()

# Create targets to make the sources show up in IDEs for convenience

# For matlab interface
if(NOT AMICI_PYTHON_BUILD_EXT_ONLY)

  set(MATLAB_SOURCES
      src/interface_matlab.cpp src/returndata_matlab.cpp
      include/amici/interface_matlab.h include/amici/returndata_matlab.h)
  find_package(Matlab)
  # In case we can find Matlab, we create a respective library to compile the
  # extension from cmake. Otherwise we just create a dummy target for the files
  # to show up inside IDEs. (Set the Matlab_ROOT_DIR cmake variable if CMake
  # cannot find your Matlab installation)
  if(${Matlab_FOUND})
    add_library(matlabInterface ${MATLAB_SOURCES})
    set_target_properties(matlabInterface PROPERTIES INCLUDE_DIRECTORIES
                                                     "${Matlab_INCLUDE_DIRS}")
    target_link_libraries(matlabInterface PUBLIC amici)
  else()
    add_custom_target(
      matlabInterface
      SOURCES ${MATLAB_SOURCES}
      COMMENT "Dummy target for MATLAB interface files")
  endif()
  set_property(
    TARGET matlabInterface
    APPEND
    PROPERTY INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/include/")
endif()

# For template files
add_custom_target(
  fileTemplates
  SOURCES src/CMakeLists.template.cmake
          src/main.template.cpp
          src/model_header.template.h
          src/model.template.cpp
          src/wrapfunctions.template.h
          src/wrapfunctions.template.cpp
          swig/CMakeLists_model.cmake
          swig/modelname.template.i
  COMMENT "Dummy target for SWIG files")
set_target_properties(
  fileTemplates PROPERTIES INCLUDE_DIRECTORIES
                           "${CMAKE_CURRENT_SOURCE_DIR}/include/")

if(NOT AMICI_PYTHON_BUILD_EXT_ONLY)
  include(clang-tools)
  include(cmakelang-tools)
endif()

set(AUTHORS "Fabian Froehlich, Jan Hasenauer, Daniel Weindl and Paul Stapor")
set(AUTHOR_EMAIL "Fabian_Froehlich@hms.harvard.edu")

# <Export cmake configuration>
install(
  TARGETS ${PROJECT_NAME}
  EXPORT AmiciTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/amici)
export(
  EXPORT AmiciTargets
  FILE AmiciTargets.cmake
  NAMESPACE Upstream::)
include(CMakePackageConfigHelpers)
include(version)
configure_package_config_file(
  "cmake/AmiciConfig.cmake.in" "${CMAKE_CURRENT_BINARY_DIR}/AmiciConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Amici")
write_basic_package_version_file(AmiciConfigVersion.cmake
                                 COMPATIBILITY ExactVersion)
install(
  EXPORT AmiciTargets
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Amici"
  NAMESPACE Upstream::)
install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/AmiciConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/AmiciConfigVersion.cmake
        ${CMAKE_CURRENT_LIST_DIR}/cmake/AmiciFindBLAS.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Amici)

install(DIRECTORY ThirdParty/gsl/gsl TYPE INCLUDE)
# When running from setup.py, this is a symlink we need to dereference
get_filename_component(_swig_realpath "swig" REALPATH)
install(DIRECTORY "${_swig_realpath}"
        DESTINATION ${CMAKE_INSTALL_DATADIR}/amici)

configure_file(cmake/AmiciFindBLAS.cmake
               ${CMAKE_CURRENT_BINARY_DIR}/AmiciFindBLAS.cmake COPYONLY)

# Register package?
if(EXPORT_PACKAGE)
  export(PACKAGE Amici)
endif()
# </Export cmake configuration>

# build interfaces for other languages
if(ENABLE_SWIG)
  add_subdirectory(swig)
endif()

if(ENABLE_PYTHON AND NOT AMICI_PYTHON_BUILD_EXT_ONLY)
  add_subdirectory(python)
endif()

if(BUILD_TESTS AND NOT AMICI_PYTHON_BUILD_EXT_ONLY)
  if(HDF5_FOUND)
    enable_testing()

    add_subdirectory(tests/cpp)
  else()
    message(WARNING "Cannot build tests without HDF5 support.")
  endif()

endif()

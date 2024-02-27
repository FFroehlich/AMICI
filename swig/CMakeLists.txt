#
# Build additional AMICI interfaces via swig
#

# Use most recent SWIG version available
set(CMAKE_FIND_PACKAGE_SORT_DIRECTION DEC)
find_package(SWIG REQUIRED)

set(SWIG_VERSION_MIN "3.0")
if(${SWIG_VERSION} VERSION_LESS ${SWIG_VERSION_MIN})
  message(FATAL_ERROR "Requiring SWIG>=${SWIG_VERSION_MIN} "
                      "but found only ${SWIG_VERSION}.")
endif()

if(DEFINED ENV{PYTHON_EXECUTABLE})
  set(Python3_EXECUTABLE $ENV{PYTHON_EXECUTABLE})
endif()

find_package(
  Python3
  COMPONENTS Interpreter Development NumPy
  REQUIRED)
set(AMICI_INTERFACE_LIST
    ${CMAKE_CURRENT_SOURCE_DIR}/amici.i
    ${CMAKE_CURRENT_SOURCE_DIR}/edata.i
    ${CMAKE_CURRENT_SOURCE_DIR}/rdata.i
    ${CMAKE_CURRENT_SOURCE_DIR}/misc.i
    ${CMAKE_CURRENT_SOURCE_DIR}/model.i
    ${CMAKE_CURRENT_SOURCE_DIR}/model_ode.i
    ${CMAKE_CURRENT_SOURCE_DIR}/model_dae.i
    ${CMAKE_CURRENT_SOURCE_DIR}/solver.i
    ${CMAKE_CURRENT_SOURCE_DIR}/solver_cvodes.i
    ${CMAKE_CURRENT_SOURCE_DIR}/solver_idas.i
    ${CMAKE_CURRENT_SOURCE_DIR}/std_unique_ptr.i
    ${CMAKE_CURRENT_SOURCE_DIR}/hdf5.i
    ${CMAKE_CURRENT_SOURCE_DIR}/abstract_model.i
    ${CMAKE_CURRENT_SOURCE_DIR}/stdvec2numpy.h)

# Add target to show files in IDE
add_custom_target(
  swigInterface
  SOURCES ${AMICI_INTERFACE_LIST}
  COMMENT "Dummy target for SWIG files.")

# https://cmake.org/cmake/help/latest/module/UseSWIG.html
include(${SWIG_USE_FILE})

set(UseSWIG_TARGET_NAME_PREFERENCE STANDARD)
set_property(SOURCE amici.i PROPERTY CPLUSPLUS ON)
set_property(SOURCE amici.i PROPERTY OUTPUT_NAME _amici)

swig_add_library(
  _amici
  LANGUAGE Python
  SOURCES amici.i
  TYPE MODULE)

set_target_properties(
  _amici
  PROPERTIES SWIG_USE_TARGET_INCLUDE_DIRECTORIES TRUE
             PREFIX ""
             BUILD_RPATH_USE_ORIGIN TRUE
             INSTALL_RPATH ThirdParty/SuiteSparse/lib
             BUILD_RPATH ThirdParty/SuiteSparse/lib)

get_target_property(AMICI_HEADERS amici PUBLIC_HEADER)
file(GLOB SWIG_FILES *.i *.h)

set_property(TARGET _amici PROPERTY SWIG_DEPENDS ${AMICI_HEADERS} ${SWIG_FILES})

# Python extension suffix
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c
          "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
  OUTPUT_VARIABLE PY_EXT_SUFFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT "${PY_EXT_SUFFIX}" STREQUAL "")
  message(STATUS "Python extension suffix is ${PY_EXT_SUFFIX}")
  set_target_properties(_amici PROPERTIES SUFFIX "${PY_EXT_SUFFIX}")
endif()

target_include_directories(
  _amici
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../include
          ${Python3_NumPy_INCLUDE_DIRS})

set_property(
  TARGET _amici
  APPEND
  PROPERTY SWIG_COMPILE_OPTIONS -threads -Wall)

if(NOT HDF5_FOUND)
  set_target_properties(_amici PROPERTIES SWIG_COMPILE_DEFINITIONS
                                          AMICI_SWIG_WITHOUT_HDF5)
endif()

if(${SWIG_VERSION} VERSION_GREATER_EQUAL 4.0.0)
  set_property(
    TARGET _amici
    APPEND
    PROPERTY SWIG_COMPILE_OPTIONS -doxygen)
else()
  set_property(
    TARGET _amici
    APPEND
    PROPERTY SWIG_COMPILE_OPTIONS -modern)
endif()
if(${SWIG_VERSION} VERSION_LESS 4.1.0)
  set_property(
    TARGET _amici
    APPEND
    PROPERTY SWIG_COMPILE_OPTIONS -py3)
endif()

# NOTE: No public definitions of any dependency are forwarded to swig, they are
# only used for compiling the swig-generated source file. Any definition that
# are relevant for swig-code generation, need to be forwarded manually.
target_link_libraries(_amici amici Python3::Python)
if(WIN32)
  add_custom_command(
    TARGET _amici
    POST_BUILD
    COMMAND dumpbin "/DEPENDENTS" "$<TARGET_FILE:_amici>"
    COMMENT "Dumping extension dependencies.")
  message("Dependencies: ${dump}")
endif()

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/amici.py $<TARGET_FILE:_amici>
        DESTINATION .)

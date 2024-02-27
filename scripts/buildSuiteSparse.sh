#!/bin/bash
#
# Build SuiteSpare
#
set -e

script_path=$(dirname "$BASH_SOURCE")
amici_path=$(cd "$script_path/.." && pwd)

suitesparse_root="${amici_path}/ThirdParty/SuiteSparse"
for subdir in SuiteSparse_config BTF AMD COLAMD KLU; do
  export CMAKE_OPTIONS="-DSUITESPARSE_USE_CUDA=OFF -DSUITESPARSE_USE_FORTRAN=OFF"

  if [ $subdir = "SuiteSparse_config" ]; then
    export CMAKE_OPTIONS="$CMAKE_OPTIONS -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DBLA_VENDOR=All -DSUITESPARSE_USE_64BIT_BLAS=ON -DBLAS_LIBRARIES=dummy"
  elif [ $subdir = "KLU" ]; then
    export CMAKE_OPTIONS="$CMAKE_OPTIONS -DKLU_USE_CHOLMOD=OFF"
  fi

  cd "${suitesparse_root}/${subdir}" && make local install
done

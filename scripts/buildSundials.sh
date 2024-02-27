#!/bin/bash
#
# Build Sundials
#
set -e

# shellcheck disable=SC2128
script_path=$(dirname "$BASH_SOURCE")
amici_path=$(cd "$script_path/.." && pwd)

suitesparse_root="${amici_path}/ThirdParty/SuiteSparse"
sundials_build_path="${amici_path}/ThirdParty/sundials/build/"

cmake=${CMAKE:-cmake}
make=${MAKE:-make}

if [[ $GITHUB_ACTIONS = true ]]; then
  # Running on CI server
  build_type="Debug"
else
  build_type="RelWithDebInfo"
fi

# enable SuperLUMT support if library exists
SuperLUMT=""
if [[ -f "${amici_path}/ThirdParty/SuperLU_MT_3.1/lib/libsuperlu_mt_PTHREAD.a" ]]
then
    SuperLUMT="-DSUPERLUMT_ENABLE=ON -DBLAS_ENABLE=ON \
         -DSUPERLUMT_INCLUDE_DIR=\"${amici_path}/ThirdParty/SuperLU_MT_3.1/SRC/\" \
         -DSUPERLUMT_LIBRARY_DIR=\"${amici_path}/ThirdParty/SuperLU_MT_3.1/lib/\""
fi

mkdir -p "${sundials_build_path}"
cd "${sundials_build_path}"

${cmake} -DCMAKE_INSTALL_PREFIX="${sundials_build_path}" \
  -DCMAKE_BUILD_TYPE="$build_type" \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DBUILD_ARKODE=OFF \
  -DBUILD_CVODE=OFF \
  -DBUILD_IDA=OFF \
  -DBUILD_KINSOL=OFF \
  -DBUILD_SHARED_LIBS=OFF \
  -DBUILD_STATIC_LIBS=ON \
  -DBUILD_NVECTOR_MANYVECTOR=OFF \
  -DBUILD_SUNNONLINSOL_PETSCSNES=OFF \
  -DEXAMPLES_ENABLE_C=OFF \
  -DEXAMPLES_INSTALL=OFF \
  -DENABLE_KLU=ON \
  -DKLU_LIBRARY_DIR="${suitesparse_root}/lib" \
  -DKLU_INCLUDE_DIR="${suitesparse_root}/include/suitesparse" \
  ${SuperLUMT} \
  ..

${make}
${make} install

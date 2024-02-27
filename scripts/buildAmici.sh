#!/bin/bash
#
# Build libamici
#
set -eou pipefail

cmake=${CMAKE:-cmake}
make=${MAKE:-make}

script_path=$(dirname "$BASH_SOURCE")
amici_path=$(cd "$script_path/.." && pwd)
amici_build_dir="${amici_path}/build"
mkdir -p "${amici_build_dir}"
cd "${amici_build_dir}"

if [ "${GITHUB_ACTIONS:-}" = true ] ||
  [ "${ENABLE_AMICI_DEBUGGING:-}" = TRUE ]; then
  # Running on CI server
  build_type="Debug"
  # exceptions instead of terminate()
  extra_cxx_flags=";-Dgsl_CONFIG_CONTRACT_VIOLATION_THROWS;-Dgsl_CONFIG_NARROW_THROWS_ON_TRUNCATION=1;-Werror;-Wno-error=deprecated-declarations"
else
  build_type="RelWithDebInfo"
  extra_cxx_flags=""
fi

# required for build swig interface
pip show numpy > /dev/null || python3 -m pip install numpy

${cmake} \
  -Wdev -DAMICI_CXX_OPTIONS="-Wall;-Wextra${extra_cxx_flags}" \
  -DCMAKE_BUILD_TYPE=$build_type \
  -DPython3_EXECUTABLE="$(command -v python3)" ..

# build, with or without sonarcloud wrapper
if [ "${CI_SONARCLOUD:-}" = "TRUE" ]; then
  build-wrapper-linux-x86-64 \
    --out-dir "${amici_path}/bw-output" \
    cmake --build . --parallel
elif [ "${GITHUB_ACTIONS:-}" = "true" ]; then
  cmake --build .
  ${make} python-sdist
else
  cmake --build . --parallel
  ${make} python-sdist
fi

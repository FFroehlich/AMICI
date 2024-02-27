#!/usr/bin/env bash
# Download and build Doxygen (in case apt or homebrew version is buggy again)
set -euo pipefail

SCRIPT_PATH=$(dirname "$BASH_SOURCE")
AMICI_PATH=$(cd "$SCRIPT_PATH"/.. && pwd)

DOXYGEN_DIR="${AMICI_PATH}"/ThirdParty/doxygen
cd "${AMICI_PATH}"/ThirdParty
if [[ ! -d ${DOXYGEN_DIR} ]]; then
  git clone --single-branch \
            --branch Release_1_10_0 \
            --depth 1 \
            -c advice.detachedHead=false \
            https://github.com/doxygen/doxygen.git "${DOXYGEN_DIR}"
fi

cd "${DOXYGEN_DIR}"
mkdir -p build
cd build
cmake -G "Unix Makefiles" ..
make -j2
make install

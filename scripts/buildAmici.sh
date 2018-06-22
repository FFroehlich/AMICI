#!/bin/bash
#
# Build libamici
#
set -e

AMICI_PATH="`dirname \"$BASH_SOURCE\"`"
AMICI_PATH="`( cd \"$AMICI_PATH/..\" && pwd )`"

mkdir -p ${AMICI_PATH}/build
cd ${AMICI_PATH}/build
CPPUTEST_BUILD_DIR=${AMICI_PATH}/ThirdParty/cpputest-master/build/
CppUTest_DIR=${CPPUTEST_BUILD_DIR} cmake -DCMAKE_BUILD_TYPE=Debug ..
make
make python-wheel

python3 -m venv ${AMICI_PATH}/build/venv --clear
${AMICI_PATH}/build/venv/bin/pip3 install --prefix= `ls -t ${AMICI_PATH}/build/python/amici-*.whl | head -1`

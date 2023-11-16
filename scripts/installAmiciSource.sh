#!/bin/bash
#
# Build libamici
#
set -e

SCRIPT_PATH=$(dirname $BASH_SOURCE)
AMICI_PATH=$(cd $SCRIPT_PATH/.. && pwd)

# Disabled until cmake package is made compatible with updated setup.py
#make python-wheel
#pip3 install --user --prefix= `ls -t ${AMICI_PATH}/build/python/amici-*.whl | head -1`

# test install from setup.py
set +e
python3 -m venv ${AMICI_PATH}/build/venv --clear
# in case this fails (usually due to missing ensurepip, try getting pip
# manually
if [[ $? ]]; then
    set -e
    python3 -m venv ${AMICI_PATH}/build/venv --clear --without-pip
    source ${AMICI_PATH}/build/venv/bin/activate
    curl https://bootstrap.pypa.io/get-pip.py -o ${AMICI_PATH}/build/get-pip.py
    python3 ${AMICI_PATH}/build/get-pip.py
else
    set -e
    source ${AMICI_PATH}/build/venv/bin/activate
fi

pip install --upgrade pip wheel
pip install --upgrade pip scipy matplotlib coverage pytest \
   pytest-cov cmake_build_extension numpy \
   git+https://github.com/ICB-DCM/fiddy.git
pip install git+https://github.com/FFroehlich/pysb@fix_pattern_matching # pin to PR for SPM with compartments
AMICI_BUILD_TEMP="${AMICI_PATH}/python/sdist/build/temp" pip install --verbose -e ${AMICI_PATH}/python/sdist[petab,test,vis] --no-build-isolation
deactivate
